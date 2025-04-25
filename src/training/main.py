import argparse
from .scripts.train_evaluator import train_evaluator
from .scripts.train_detector import train_detector

def main():
    parser = argparse.ArgumentParser(description="Train protein models")
    parser.add_argument("--model", choices=["evaluator", "detector", "both"], 
                      default="both", help="Which model to train")
    
    args = parser.parse_args()
    
    if args.model == "evaluator" or args.model == "both":
        print("=== Training Evaluator Model ===")
        train_evaluator()
    
    if args.model == "detector" or args.model == "both":
        print("=== Training Detector Model ===")
        train_detector()
    
    print("Training completed!")

if __name__ == "__main__":
    main()