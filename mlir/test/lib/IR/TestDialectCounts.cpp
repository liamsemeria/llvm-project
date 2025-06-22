#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <map>
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {
    struct TestDialectCountsPass : public PassWrapper<TestDialectCountsPass, OperationPass<>> {
    
    std::map<StringRef, int> dialect_counts;

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDialectCountsPass)

        StringRef getArgument() const final { return "test-dialect-counts"; }
        StringRef getDescription() const final { return "count ops of each dialect."; }

        void runOnOperation() override {
            // Get the current operation being operated on.
            Operation *op = getOperation();
            run(op);

            for (auto iter : dialect_counts) {
                llvm::outs() << iter.first << " : " << iter.second << "\n";
            }
        }

        void run(Operation *op) {
            // count the dialect for the operation
            StringRef s = op->getDialect()->getNamespace();
            if (dialect_counts.find(s) == dialect_counts.end())
                dialect_counts[s] = 1;
            else
                dialect_counts[s]++;
            
            // traverse operations
            for (Region &r : op->getRegions()) {
                for (Block &b : r.getBlocks()) {
                    for (Operation &o : b.getOperations()) {
                        run(&o);
                    }
                }
            }
        }
    };
}

namespace mlir {
    void registerTestDialectCountsPass() {
        PassRegistration<TestDialectCountsPass>();
    }
}