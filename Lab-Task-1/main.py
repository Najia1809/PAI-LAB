import pandas as pd
from scrapper import extract_emails

# Read Excel file
input_file = "urls.xlsx"
df = pd.read_excel(input_file)

# Make sure your Excel has a column named 'URL'
if 'URL' not in df.columns:
    print("Error: Excel file must have a column named 'URL'")
    exit()

# Create a new column for emails
df['Emails'] = ""

# Loop through each URL and extract emails
for index, row in df.iterrows():
    url = row['URL']
    print(f"Processing: {url}")
    emails = extract_emails(url)
    if isinstance(emails, set):
        df.at[index, 'Emails'] = ", ".join(emails) if emails else "No emails found"
    else:
        df.at[index, 'Emails'] = emails  # Error message

# Save results to a new Excel file
output_file = r"C:\Users\gold.lab10\Desktop\WEB-SCRAPPER\emails_output.xlsx"
try:
    df.to_excel(output_file, index=False)
    print(f"Results saved in {output_file}")
except Exception as e:
    print(f"Failed to save Excel: {e}")


