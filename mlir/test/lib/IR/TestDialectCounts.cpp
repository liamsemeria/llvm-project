#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <map>
#include <string>
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Option.h"

using namespace mlir;

namespace {
    struct TestDialectCountsPass : public PassWrapper<TestDialectCountsPass, OperationPass<>> {
    


    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDialectCountsPass)

        Option<std::string> printMsg{*this, "print-msg", llvm::cl::desc("if specified will print before colors")};

        TestDialectCountsPass() = default;
        TestDialectCountsPass(const TestDialectCountsPass& p) : PassWrapper(p) {}

        //Option<bool> includeColorKey{*this, "include_color_key", llvm::cl::desc("wether or not to print the color key after printing")}
        

        StringRef getArgument() const final { return "test-dialect-counts"; }
        StringRef getDescription() const final { return "count ops of each dialect."; }

        void runOnOperation() override {
            if (! printMsg.empty()) 
                llvm::outs() << printMsg << "\n";

            // Get the current operation being operated on.
            Operation *op = getOperation();
            std::map<StringRef, int> dialect_counts;
            std::map<StringRef, std::string> dialect_colors;

            run(op, dialect_counts, dialect_colors);

            // print the color key
            llvm::outs() << "\n";
            for (auto iter : dialect_counts) {
                llvm::outs() << dialect_colors[iter.first] << iter.first << ": " << iter.second << " ";
            }
            llvm::outs() << "\n";
        }

        void run(Operation *op,std::map<StringRef, int>& dialect_counts, std::map<StringRef, std::string>& dialect_colors) {
            // count the dialect for the operation
            StringRef s = op->getDialect()->getNamespace();
            

            if (dialect_counts.find(s) == dialect_counts.end()) {
                dialect_counts[s] = 1;
                // get color hash and create format string
                std::string suffix = "m \033[0m";
                std::string f = "\033[48;5;" + std::to_string((int)(llvm::hash_value(s) % 230 + 1)) + suffix;
                dialect_colors[s] = f;
            } else
                dialect_counts[s]++;
            
            llvm::outs() << dialect_colors[s];
            
            // traverse operations
            for (Region &r : op->getRegions()) {
                for (Block &b : r.getBlocks()) {
                    for (Operation &o : b.getOperations()) {
                        run(&o, dialect_counts, dialect_colors);
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