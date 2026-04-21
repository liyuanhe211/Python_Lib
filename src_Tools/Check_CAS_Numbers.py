import re

def validate_cas_number(cas_input):
    """
    Validates a CAS Registry Number.
    Returns a tuple (is_valid, message).
    """
    cas_str = cas_input.strip()
    
    # Basic format check using regex
    # 1st group: 2 to 7 digits
    # 2nd group: exactly 2 digits
    # 3rd group: exactly 1 digit
    pattern = r'^\d{2,7}-\d{2}-\d$'
    if not re.match(pattern, cas_str):
        return False, "Invalid format"

    # Split into parts to calculate check digit
    parts = cas_str.split('-')
    
    # The last digit is the check digit
    check_digit_str = parts[-1]
    check_digit = int(check_digit_str)
    
    # Combine the first two parts to get the digits for calculation
    # Example: 7732-18-5 -> "773218"
    digits_str = parts[0] + parts[1]
    
    # Calculate sum
    # Starting from the right-most remaining digit, multiply by 1, 2, 3...
    total_sum = 0
    for i, digit in enumerate(reversed(digits_str)):
        weight = i + 1
        total_sum += int(digit) * weight
        
    # Calculate expected check digit
    calculated_check_digit = total_sum % 10
    
    if calculated_check_digit == check_digit:
        return True, "Valid"
    else:
        return False, f"Invalid check digit (Expected {calculated_check_digit}, got {check_digit})"

def main():
    list_to_check = r'''10193-95-0
9036-19-5 
90-13-1
75-12-7
121-65-3
112-04-9
-
39420-45-6
25322-68-3
-
33007-83-9
9002-98-6
9003-20-7
7757-82-6
7647-14-5
25014-41-9
9002-98-6
1314-13-2
1346753-09-0
9003-53-6
-
7631-99-4
2348-82-5
24772-63-2
141-53-7
96-49-1
7778-18-9, 7646-79-9
9011-14-7 
66-22-8
7601-89-0 
50-70-4
9011-14-7
50-70-4
17927-65-0
100-97-0
30525-89-4
52829-07-9
25704-18-1
6035-47-8
25014-41-9
302818-73-1
150-13-0
195456-48-5
'''

    print(f"{'CAS Number':<15} | {'Status':<10} | {'Message'}")
    print("-" * 45)

    # Process each line
    for line in list_to_check.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that are clearly not CAS numbers (like just hyphens) if desired,
        # but the prompt implies checking the list provided.
        # We will try to validate everything that looks remotely like a string.
        
        is_valid, message = validate_cas_number(line)
        status = "VALID" if is_valid else "INVALID"
        print(f"{line:<15} | {status:<10} | {message}")

if __name__ == "__main__":
    main()
