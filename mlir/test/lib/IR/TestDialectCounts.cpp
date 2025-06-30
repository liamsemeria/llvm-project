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
                llvm::outs() << "\n" << printMsg << "\n";

            // walk starting at the main op
            getOperation()->walk([this](Operation *o) {
                run(o);
            });

            // print the color key
            llvm::outs() << "\n";
            for (auto iter : dialectCounts) {
                llvm::outs() << dialectColors[iter.first] << iter.first << ": " << iter.second << " ";
            }
            llvm::outs() << "\n";
        }
        private:
        std::map<StringRef, int> dialectCounts;
        std::map<StringRef, std::string> dialectColors;

        void run(Operation *op) {
            // count the dialect for the operation
            StringRef s = op->getDialect()->getNamespace();
            

            if (dialectCounts.find(s) == dialectCounts.end()) {
                dialectCounts[s] = 1;
                int offset = 0;
                // get color hash and create format string
                std::string f = getDialectColor(s);
                // make sure the color is unique, if not, rehash
                bool clone = true;
                while (clone) {
                    for (auto iter : dialectColors) {
                        if (f == iter.second) {
                            f = getDialectColor(s,offset++);
                            break;
                        }
                    }
                    clone = false;
                }
                dialectColors[s] = f;
            } else
                dialectCounts[s]++;
            
            llvm::outs() << dialectColors[s];
        }

        // get the color for a dialect, set offset to nonzero if a rehash is needed
        std::string getDialectColor(StringRef dialectName, int offset = 0) {
            return "\033[48;5;" + std::to_string((int)((llvm::hash_value(dialectName) + offset) % 230 + 1)) + "m \033[0m";
        }
    };
}

namespace mlir {
    void registerTestDialectCountsPass() {
        PassRegistration<TestDialectCountsPass>();
    }
}