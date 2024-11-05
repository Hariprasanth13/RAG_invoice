import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker and some example data
faker = Faker()
currencies = ["$", "€", "£"]
parts = ["Widget A", "Gearbox Assembly", "Hydraulic Pump", "Circuit Board", "Bearing Set", "Control Valve", "LED Display", "Air Filter", "Power Supply Unit", "Optical Sensor"]
vendors = ["ABC Corporation", "XYZ Ltd.", "Tech Solutions Ltd.", "Fournisseur Global", "GlobalTech Inc.", "ABC Enterprises", "Quick Solutions", "ABC Services", "Digital Works", "New Horizons LLC"]
buyers = [("John Doe", 101), ("Alice Smith", 102), ("Bob Johnson", 103), ("Claire Lee", 104), ("David Wilson", 105), ("Emma Davis", 106), ("Michael Brown", 107), ("Olivia Green", 108), ("Henry White", 109), ("Sophia Harris", 110)]

# Helper function to generate a random date
def random_date(start_date, end_date):
    return start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

# Generate synthetic data
data = []

for i in range(100):
    # Random selection for each invoice
    invoice_number = faker.random_number(digits=5)
    vendor_name = random.choice(vendors)
    issue_date = random_date(datetime(2024, 10, 1), datetime(2024, 10, 9)).strftime("%Y-%m-%d")
    amount = f"{random.choice(currencies)}{random.randint(100, 5000)}"
    part_description = random.choice(parts)
    reception_date = random_date(datetime(2024, 10, 1), datetime(2024, 10, 9)).strftime("%Y-%m-%d")
    buyer = random.choice(buyers)
    
    # Create an invoice text
    invoice_text = (
        f"Invoice No. {invoice_number} issued by {vendor_name} on {issue_date}. "
        f"Amount Due: {amount}. Part: {part_description}, Received: {reception_date}. "
        f"Buyer: ID {buyer[1]}, {buyer[0]}."
    )
    
    # Create annotations
    annotation = {
        "invoice_number": str(invoice_number),
        "vendor_name": vendor_name,
        "date": issue_date,
        "total_amount": amount,
        "part_description": part_description,
        "date_of_reception": reception_date,
        "buyer_id": str(buyer[1]),
        "buyer_name": buyer[0]
    }

    data.append({"invoice_text": invoice_text, "annotation": annotation})

# Output the first 5 examples
for item in data[:5]:
    print(f"Invoice Text: {item['invoice_text']}")
    print(f"Annotation: {item['annotation']}")
    print()
