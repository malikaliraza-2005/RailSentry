# Student name: [Your Name]
# Roll number: [Your Roll Number]
# Course code: [Your Course Code]
# Module description: Main menu for RailSentry

"""
Main application for RailSentry.
It connects all four project modules.
"""

from pathlib import Path

try:
    import module1_agent_search
    import module2_data_ml
    import module3_neural_network
    import module4_integrated_system
except ImportError as error:
    print("A module could not be imported:")
    print(error)
    print("Make sure main.py and all module files are in the same folder.")
    raise


PROJECT_DIR = Path(__file__).resolve().parent


def print_menu() -> None:
    """Display the main application menu."""
    print("\n" + "=" * 55)
    print("RailSentry")
    print("AI-Based Railway Station Incident Response System")
    print("=" * 55)
    print("1. Agent and Search Algorithms")
    print("2. Data Analysis and Machine Learning")
    print("3. Neural Network Training and Evaluation")
    print("4. Integrated Smart System")
    print("5. Run Complete Demonstration")
    print("6. Exit")


def run_module_safely(module_name: str, function) -> None:
    """Run one module while preventing one error from closing the app."""
    try:
        print(f"\nRunning {module_name}...")
        function()
    except KeyboardInterrupt:
        print("\nModule stopped by the user.")
    except Exception as error:
        print(f"\nAn error occurred in {module_name}:")
        print(error)


def run_complete_demo() -> None:
    """Run all four modules one after another."""
    print("\nCOMPLETE PROJECT DEMONSTRATION")

    run_module_safely(
        "Module 1",
        module1_agent_search.run_module1,
    )
    run_module_safely(
        "Module 2",
        module2_data_ml.run_module2,
    )
    run_module_safely(
        "Module 3",
        module3_neural_network.run_module3,
    )
    run_module_safely(
        "Module 4",
        module4_integrated_system.run_module4,
    )

    print("\nComplete demonstration finished.")


def main() -> None:
    """Start the menu-driven application."""
    while True:
        print_menu()

        try:
            choice = input("Enter your choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting application.")
            break

        if choice == "1":
            run_module_safely(
                "Module 1",
                module1_agent_search.module1_menu,
            )
        elif choice == "2":
            run_module_safely(
                "Module 2",
                module2_data_ml.module2_menu,
            )
        elif choice == "3":
            run_module_safely(
                "Module 3",
                module3_neural_network.module3_menu,
            )
        elif choice == "4":
            run_module_safely(
                "Module 4",
                module4_integrated_system.module4_menu,
            )
        elif choice == "5":
            run_complete_demo()
        elif choice == "6":
            print("Thank you. The application is now closed.")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 6.")


if __name__ == "__main__":
    print("Project folder:", PROJECT_DIR)
    main()
