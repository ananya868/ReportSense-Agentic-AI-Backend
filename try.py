def medicine_ocr(img_path):
    """
    Perform OCR on image to extract medicine names.
    
    Args:
        img_path (str): Path to the image containing medicine information
        
    Returns:
        list: List of detected medicine names, empty list if none found
    """
    # OCR implementation would go here
    return []  # Currently returns empty dict as placeholder


def get_manual_entry():
    """
    Get medicine names through manual user input.
    
    Returns:
        list: List of medicine names entered by user
    """
    while True:
        entry_type = input("Do you want to enter one medicine name or multiple? (one/multiple): ").strip().lower()
        if entry_type not in ["one", "multiple"]:
            print("Invalid choice. Please enter 'one' or 'multiple'.")
            continue
        
        user_input = input("Enter medicine name(s), use commas if entering multiple: ")
        if entry_type == "multiple" and "," not in user_input:
            print("You selected multiple but did not use commas. Please separate names correctly.")
            continue
        
        medicines = [med.strip() for med in user_input.split(",") if med.strip()]
        return medicines


def confirm_medicines(medicines):
    """
    Ask user to confirm the detected/entered medicine names.
    
    Args:
        medicines (list): List of medicine names to confirm
        
    Returns:
        tuple: (bool, list) - (Whether medicines are confirmed, confirmed medicine list)
    """
    if not medicines:
        return False, []
        
    print(f"Please confirm the following medicines: {', '.join(medicines)}")
    confirmation = input("Are these correct? (yes/no): ").strip().lower()
    
    return confirmation == "yes", medicines


def process_medicines():
    """
    Main function to handle medicine detection workflow.
    
    Returns:
        list: Final list of confirmed medicines
    """
    img_path = "pharmacist_agent/medi_data/med.png"
    medicines = []
    confirmed = False
    
    while not confirmed:
        method = input("Please select method to fetch medicine names: \n 1. OCR \n 2. Manual Entry \n:")
        
        if method == "1":
            medicines = medicine_ocr(img_path)
            if not medicines:
                print(
                    "No medicines found! Please select one out of the following: \n",
                    "1. Enter medicine(s) name manually \n",
                    "2. Retry the OCR process \n",
                )
                choice = input("Please select an option (1 or 2): ")
                if choice == "1":
                    medicines = get_manual_entry()
                elif choice == "2":
                    medicines = medicine_ocr(img_path)
        
        elif method == "2":
            medicines = get_manual_entry()
        
        else:
            print("Invalid option. Please select 1 or 2.")
            continue
            
        if not medicines:
            print("No valid medicines found!")
            retry = input("Would you like to try again? (yes/no): ").strip().lower()
            if retry != "yes":
                return []
            continue
            
        confirmed, medicines = confirm_medicines(medicines)
        
        if not confirmed:
            print("Medicine names not confirmed. Let's try again.")
    
    print(f"Final medicines confirmed: {medicines}")
    return medicines


if __name__ == "__main__":
    final_medicines = process_medicines()
    if final_medicines:
        print(f"Processing will continue with these medicines: {final_medicines}")
    else:
        print("Process terminated without valid medicines.")