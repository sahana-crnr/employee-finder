import sys
from src.agent import run_agent

def main():
    print("LangChain AI Employee Finder Agent")
    print("-" * 60)

    try:
        company = input("\nEnter company name: ").strip()
        if not company:
            print("Please enter a valid company name.")
            return

        role = input("Enter target role/department (optional, press Enter to skip): ").strip()
        role_preference = role if role else None

        run_agent(company, role_preference)

    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
