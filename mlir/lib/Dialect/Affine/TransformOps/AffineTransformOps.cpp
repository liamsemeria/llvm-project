//=== AffineTransformOps.cpp - Implementation of Affine transformation ops ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
TileOp::apply(transform::TransformRewriter &rewriter, TransformResults &results,
              TransformState &state) {
  auto payload = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payload))
    return emitSilenceableError()
           << "expected exactly one affine.for target, found "
           << llvm::range_size(payload);

  auto target = dyn_cast<AffineForOp>(*payload.begin());
  if (!target) {
    auto diag = emitSilenceableError()
                << "expected target to be an affine.for operation";
    diag.attachNote((*payload.begin())->getLoc()) << "target payload op";
    return diag;
  }

  SmallVector<AffineForOp> nest;
  getPerfectlyNestedLoops(nest, target);
  if (getDimensions().size() != getTileSizes().size()) {
    auto diag = emitSilenceableError()
                << "expected dimensions and tile sizes to have equal lengths, "
                   "found "
                << getDimensions().size() << " and " << getTileSizes().size();
    diag.attachNote(target.getLoc()) << "target payload op";
    return diag;
  }

  SmallVector<unsigned> dimensions;
  SmallVector<unsigned> tileSizes;
  dimensions.reserve(getDimensions().size());
  tileSizes.reserve(getTileSizes().size());
  for (auto [dimension, tileSize] :
       llvm::zip_equal(getDimensions(), getTileSizes())) {
    if (dimension < 0 || static_cast<uint64_t>(dimension) >= nest.size()) {
      auto diag = emitSilenceableError()
                  << "dimension " << dimension
                  << " is out of range for an affine loop nest of depth "
                  << nest.size();
      diag.attachNote(target.getLoc()) << "target payload op";
      return diag;
    }
    if (tileSize <= 0 || static_cast<uint64_t>(tileSize) >
                             std::numeric_limits<unsigned>::max()) {
      auto diag = emitSilenceableError()
                  << "expected positive tile sizes representable as unsigned";
      diag.attachNote(target.getLoc()) << "target payload op";
      return diag;
    }
    dimensions.push_back(static_cast<unsigned>(dimension));
    tileSizes.push_back(static_cast<unsigned>(tileSize));
  }

  SmallVector<unsigned> pointDimensions;
  std::optional<ArrayRef<int64_t>> inputPointDimensions = getPointDimensions();
  if (inputPointDimensions) {
    pointDimensions.reserve(inputPointDimensions->size());
    for (int64_t dimension : *inputPointDimensions) {
      if (dimension < 0 || static_cast<uint64_t>(dimension) >= nest.size()) {
        auto diag = emitSilenceableError()
                    << "point dimension " << dimension
                    << " is out of range for an affine loop nest of depth "
                    << nest.size();
        diag.attachNote(target.getLoc()) << "target payload op";
        return diag;
      }
      pointDimensions.push_back(static_cast<unsigned>(dimension));
    }
    SmallVector<unsigned> sortedPointDimensions(pointDimensions);
    llvm::sort(sortedPointDimensions);
    if (pointDimensions.size() != nest.size() ||
        llvm::any_of(llvm::enumerate(sortedPointDimensions),
                     [](auto entry) {
                       return entry.index() != entry.value();
                     })) {
      auto diag = emitSilenceableError()
                  << "expected point_dimensions to be a permutation of all "
                     "source dimensions";
      diag.attachNote(target.getLoc()) << "target payload op";
      return diag;
    }
  }

  if (llvm::any_of(
          nest, [](AffineForOp loop) { return loop.getNumResults() != 0; })) {
    auto diag = emitSilenceableError()
                << "affine loop tiling does not support loops with results";
    diag.attachNote(target.getLoc()) << "target payload op";
    return diag;
  }
  if (failed(checkTilePerfectlyNestedOrderedPreconditions(
          nest, dimensions, tileSizes, pointDimensions))) {
    auto diag = emitSilenceableError()
                << "affine loop tiling requires a hyperrectangular iteration "
                   "domain and representable tiled steps";
    diag.attachNote(target.getLoc()) << "target payload op";
    return diag;
  }
  if (!dimensions.empty()) {
    bool hasUnmodeledMemoryEffects = false;
    target.walk([&](Operation *nested) {
      if (isa<AffineReadOpInterface, AffineWriteOpInterface>(nested) ||
          isa<AffineForOp, AffineIfOp, AffineYieldOp>(nested) ||
          isMemoryEffectFree(nested))
        return;
      hasUnmodeledMemoryEffects = true;
    });
    if (hasUnmodeledMemoryEffects) {
      auto diag = emitSilenceableError()
                  << "affine loop tiling cannot prove legality for non-affine "
                     "memory operations";
      diag.attachNote(target.getLoc()) << "target payload op";
      return diag;
    }
    if (!isTilePerfectlyNestedOrderedValid(nest, dimensions, tileSizes,
                                            pointDimensions)) {
      auto diag = emitSilenceableError()
                  << "affine loop nest is not legal to tile in the requested "
                     "dimension order";
      diag.attachNote(target.getLoc()) << "target payload op";
      return diag;
    }
  }
  SmallVector<AffineForOp> tiledNest;
  if (failed(tilePerfectlyNestedOrdered(nest, dimensions, tileSizes,
                                        pointDimensions, &tiledNest,
                                        &rewriter)))
    return emitDefiniteFailure() << "failed to tile prevalidated affine nest";
  unsigned numTileLoops = dimensions.size();
  results.set(cast<OpResult>(getTileLoops()),
              ArrayRef(tiledNest).take_front(numTileLoops));
  results.set(cast<OpResult>(getPointLoops()),
              ArrayRef(tiledNest).drop_front(numTileLoops));
  return DiagnosedSilenceableFailure::success();
}

void TileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// AffineVectorizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure AffineVectorizeOp::apply(
    transform::TransformRewriter &rewriter, TransformResults &results,
    TransformState &state) {
  SmallVector<AffineForOp> roots;
  DenseSet<Operation *> uniqueRoots;
  for (Operation *payload : state.getPayloadOps(getTarget())) {
    auto root = dyn_cast<AffineForOp>(payload);
    if (!root)
      return emitSilenceableError()
             << "expected target payload operations to be affine.for";
    if (!uniqueRoots.insert(payload).second)
      return emitSilenceableError() << "duplicate affine point-band target";
    roots.push_back(root);
  }
  if (roots.empty())
    return emitSilenceableError() << "expected at least one affine point band";

  for (auto [index, root] : llvm::enumerate(roots)) {
    for (AffineForOp other : ArrayRef(roots).drop_front(index + 1)) {
      if (!root->isAncestor(other) && !other->isAncestor(root))
        continue;
      auto diag = emitSilenceableError()
                  << "affine vectorization targets must not overlap";
      diag.attachNote(root.getLoc()) << "first overlapping point band";
      diag.attachNote(other.getLoc()) << "second overlapping point band";
      return diag;
    }
  }

  SmallVector<AffineForOp> selectedLoops;
  DenseSet<Operation *> uniqueSelectedLoops;
  for (Operation *payload : state.getPayloadOps(getVectorizedLoops())) {
    auto loop = dyn_cast<AffineForOp>(payload);
    if (!loop)
      return emitSilenceableError()
             << "expected selected payload operations to be affine.for";
    if (!uniqueSelectedLoops.insert(payload).second)
      return emitSilenceableError() << "duplicate selected affine loop";
    selectedLoops.push_back(loop);
  }
  if (selectedLoops.empty())
    return emitSilenceableError()
           << "expected at least one selected affine loop";

  SmallVector<int64_t> vectorSizes;
  if (getVectorSizes())
    llvm::append_range(vectorSizes, *getVectorSizes());
  if (llvm::any_of(vectorSizes, [](int64_t size) { return size <= 0; }))
    return emitSilenceableError() << "expected positive vector sizes";

  SmallVector<SmallVector<AffineForOp>> selectedByRoot(roots.size());
  for (AffineForOp selected : selectedLoops) {
    std::optional<unsigned> owner;
    for (auto [rootIndex, root] : llvm::enumerate(roots)) {
      if (root != selected && !root->isAncestor(selected))
        continue;
      if (owner)
        return emitDefiniteFailure()
               << "selected loop belongs to overlapping point bands";
      owner = rootIndex;
    }
    if (!owner) {
      auto diag = emitSilenceableError()
                  << "selected affine loop does not belong to a targeted "
                     "point band";
      diag.attachNote(selected.getLoc()) << "selected loop";
      return diag;
    }
    selectedByRoot[*owner].push_back(selected);
  }

  for (auto [root, selected] : llvm::zip_equal(roots, selectedByRoot)) {
    if (selected.empty()) {
      auto diag = emitSilenceableError()
                  << "each targeted point band must contain a selected loop";
      diag.attachNote(root.getLoc()) << "point band without selected loop";
      return diag;
    }
    if (failed(checkVectorizeAffineLoopNestSelectedPreconditions(
            root, selected, vectorSizes))) {
      auto diag = emitSilenceableError()
                  << "selected affine loops are not legal to vectorize";
      diag.attachNote(root.getLoc()) << "target point band";
      return diag;
    }
  }

  for (auto [root, selected] : llvm::zip_equal(roots, selectedByRoot))
    if (failed(vectorizeAffineLoopNestSelected(root, selected, vectorSizes,
                                               &rewriter)))
      return emitSilenceableError()
             << "failed to vectorize prevalidated affine point band";
  return DiagnosedSilenceableFailure::success();
}

void AffineVectorizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getVectorizedLoopsMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ParallelizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
ParallelizeOp::apply(transform::TransformRewriter &rewriter,
                     TransformResults &results, TransformState &state) {
  SmallVector<AffineForOp> targets;
  DenseSet<Operation *> uniqueTargets;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto forOp = dyn_cast<AffineForOp>(target);
    if (!forOp) {
      auto diag = emitSilenceableError()
                  << "expected target to be an affine.for operation";
      diag.attachNote(target->getLoc()) << "target payload op";
      return diag;
    }
    if (!uniqueTargets.insert(target).second) {
      auto diag = emitSilenceableError() << "duplicate target operation";
      diag.attachNote(target->getLoc()) << "duplicate target";
      return diag;
    }
    if (forOp.getNumIterOperands() != 0) {
      auto diag = emitSilenceableError()
                  << "affine.for loops with iter arguments are not supported";
      diag.attachNote(target->getLoc()) << "target payload op";
      return diag;
    }
    if (!isLoopParallel(forOp)) {
      auto diag = emitSilenceableError()
                  << "could not prove that the affine.for loop is parallel";
      diag.attachNote(target->getLoc()) << "target payload op";
      return diag;
    }
    targets.push_back(forOp);
  }

  for (auto [index, target] : llvm::enumerate(targets)) {
    for (AffineForOp other : ArrayRef(targets).drop_front(index + 1)) {
      if (!target->isAncestor(other) && !other->isAncestor(target))
        continue;
      auto diag = emitSilenceableError() << "overlapping target operations";
      diag.attachNote(target.getLoc()) << "first overlapping target";
      diag.attachNote(other.getLoc()) << "second overlapping target";
      return diag;
    }
  }

  SmallVector<Operation *> parallelLoops;
  parallelLoops.reserve(targets.size());
  for (AffineForOp target : targets) {
    AffineParallelOp parallel;
    if (failed(affineParallelize(target, {}, &parallel, &rewriter)))
      return emitDefiniteFailure() << "failed to parallelize affine.for loop";
    parallelLoops.push_back(parallel);
  }
  results.set(cast<OpResult>(getParallel()), parallelLoops);
  return DiagnosedSilenceableFailure::success();
}

void ParallelizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// SimplifyBoundedAffineOpsOp
//===----------------------------------------------------------------------===//

LogicalResult SimplifyBoundedAffineOpsOp::verify() {
  if (getLowerBounds().size() != getBoundedValues().size())
    return emitOpError() << "incorrect number of lower bounds, expected "
                         << getBoundedValues().size() << " but found "
                         << getLowerBounds().size();
  if (getUpperBounds().size() != getBoundedValues().size())
    return emitOpError() << "incorrect number of upper bounds, expected "
                         << getBoundedValues().size() << " but found "
                         << getUpperBounds().size();
  return success();
}

namespace {
/// Simplify affine.min / affine.max ops with the given constraints. They are
/// either rewritten to affine.apply or left unchanged.
template <typename OpTy>
struct SimplifyAffineMinMaxOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  SimplifyAffineMinMaxOp(MLIRContext *ctx,
                         const FlatAffineValueConstraints &constraints,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(ctx, benefit), constraints(constraints) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    FailureOr<AffineValueMap> simplified =
        simplifyConstrainedMinMaxOp(op, constraints);
    if (failed(simplified))
      return failure();
    rewriter.replaceOpWithNewOp<AffineApplyOp>(op, simplified->getAffineMap(),
                                               simplified->getOperands());
    return success();
  }

  const FlatAffineValueConstraints &constraints;
};
} // namespace

DiagnosedSilenceableFailure
SimplifyBoundedAffineOpsOp::apply(transform::TransformRewriter &rewriter,
                                  TransformResults &results,
                                  TransformState &state) {
  // Get constraints for bounded values.
  SmallVector<int64_t> lbs;
  SmallVector<int64_t> ubs;
  SmallVector<Value> boundedValues;
  DenseSet<Operation *> boundedOps;
  for (const auto &it : llvm::zip_equal(getBoundedValues(), getLowerBounds(),
                                        getUpperBounds())) {
    Value handle = std::get<0>(it);
    for (Operation *op : state.getPayloadOps(handle)) {
      if (op->getNumResults() != 1 || !op->getResult(0).getType().isIndex()) {
        auto diag =
            emitDefiniteFailure()
            << "expected bounded value handle to point to one or multiple "
               "single-result index-typed ops";
        diag.attachNote(op->getLoc()) << "multiple/non-index result";
        return diag;
      }
      boundedValues.push_back(op->getResult(0));
      boundedOps.insert(op);
      lbs.push_back(std::get<1>(it));
      ubs.push_back(std::get<2>(it));
    }
  }

  // Build constraint set.
  FlatAffineValueConstraints cstr;
  for (const auto &it : llvm::zip(boundedValues, lbs, ubs)) {
    unsigned pos;
    if (!cstr.findVar(std::get<0>(it), &pos))
      pos = cstr.appendSymbolVar(std::get<0>(it));
    cstr.addBound(presburger::BoundType::LB, pos, std::get<1>(it));
    // Note: addBound bounds are inclusive, but specified UB is exclusive.
    cstr.addBound(presburger::BoundType::UB, pos, std::get<2>(it) - 1);
  }

  // Transform all targets.
  SmallVector<Operation *> targets;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    if (!isa<AffineMinOp, AffineMaxOp>(target)) {
      auto diag = emitDefiniteFailure()
                  << "target must be affine.min or affine.max";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    if (boundedOps.contains(target)) {
      auto diag = emitDefiniteFailure()
                  << "target op result must not be constrained";
      diag.attachNote(target->getLoc()) << "target/constrained op";
      return diag;
    }
    targets.push_back(target);
  }
  RewritePatternSet patterns(getContext());
  // Canonicalization patterns are needed so that affine.apply ops are composed
  // with the remaining affine.min/max ops.
  AffineMaxOp::getCanonicalizationPatterns(patterns, getContext());
  AffineMinOp::getCanonicalizationPatterns(patterns, getContext());
  patterns.insert<SimplifyAffineMinMaxOp<AffineMinOp>,
                  SimplifyAffineMinMaxOp<AffineMaxOp>>(getContext(), cstr);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  // Apply the simplification pattern to a fixpoint.
  if (failed(applyOpPatternsGreedily(
          targets, frozenPatterns,
          GreedyRewriteConfig()
              .setListener(
                  static_cast<RewriterBase::Listener *>(rewriter.getListener()))
              .setStrictness(GreedyRewriteStrictness::ExistingAndNewOps)))) {
    auto diag = emitDefiniteFailure()
                << "affine.min/max simplification did not converge";
    return diag;
  }
  return DiagnosedSilenceableFailure::success();
}

void SimplifyBoundedAffineOpsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  for (OpOperand &operand : getBoundedValuesMutable())
    onlyReadsHandle(operand, effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// SuperVectorizeOp
//===----------------------------------------------------------------------===//

LogicalResult SuperVectorizeOp::verify() {
  if (getFastestVaryingPattern().has_value()) {
    if (getFastestVaryingPattern()->size() != getVectorSizes().size())
      return emitOpError()
             << "fastest varying pattern specified with different size than "
                "the vector size";
  }
  return success();
}

DiagnosedSilenceableFailure
SuperVectorizeOp::apply(transform::TransformRewriter &rewriter,
                        TransformResults &results, TransformState &state) {
  ArrayRef<int64_t> fastestVaryingPattern;
  if (getFastestVaryingPattern().has_value())
    fastestVaryingPattern = getFastestVaryingPattern().value();

  for (Operation *target : state.getPayloadOps(getTarget()))
    if (!target->getParentOfType<affine::AffineForOp>())
      vectorizeChildAffineLoops(target, getVectorizeReductions(),
                                getVectorSizes(), fastestVaryingPattern);

  return DiagnosedSilenceableFailure::success();
}

void SuperVectorizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// SimplifyMinMaxAffineOpsOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
SimplifyMinMaxAffineOpsOp::apply(transform::TransformRewriter &rewriter,
                                 TransformResults &results,
                                 TransformState &state) {
  SmallVector<Operation *> targets;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    if (!isa<AffineMinOp, AffineMaxOp>(target)) {
      auto diag = emitDefiniteFailure()
                  << "target must be affine.min or affine.max";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    targets.push_back(target);
  }
  bool modified = false;
  if (failed(mlir::affine::simplifyAffineMinMaxOps(rewriter, targets,
                                                   &modified))) {
    return emitDefiniteFailure()
           << "affine.min/max simplification did not converge";
  }
  if (!modified) {
    return emitSilenceableError()
           << "the transform failed to simplify any of the target operations";
  }
  return DiagnosedSilenceableFailure::success();
}

void SimplifyMinMaxAffineOpsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class AffineTransformDialectExtension
    : public transform::TransformDialectExtension<
          AffineTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineTransformDialectExtension)

  using Base::Base;

  void init() {
    declareGeneratedDialect<AffineDialect>();
    declareGeneratedDialect<vector::VectorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.cpp.inc"

void mlir::affine::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<AffineTransformDialectExtension>();
}
