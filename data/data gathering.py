import os
import pandas as pd
import subprocess
import zipfile

# 1. Set up Kaggle Credentials
# Using the key you provided
os.environ['KAGGLE_USERNAME'] = "ryan khojikian" # Your username from the json
os.environ['KAGGLE_KEY'] = "beac6424cea98770cba823d33e11e34e"

print("🚀 Starting cloud-to-cloud download...")


# 2. Download the dataset directly from Kaggle (using subprocess for local execution)

try:
    subprocess.run([
        "kaggle", "datasets", "download", "-d", "thoughtvector/customer-support-on-twitter"
    ], check=True)
except Exception as e:
    print(f"❌ Error downloading dataset: {e}")


# 3. Unzip the downloaded file (using subprocess for local execution)
zip_path = 'customer-support-on-twitter.zip'
extract_path = 'dataset_extracted'
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Unzipped dataset successfully.")
except Exception as e:
    print(f"❌ Error unzipping file: {e}")

print("✅ Download and Extraction complete!")

# 4. Find the CSV and load it
extract_path = 'dataset_extracted'
csv_path = None
df_tickets = ''
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith('.csv'):
            csv_path = os.path.join(root, file)
            break

if csv_path:
    # Load the data
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} total rows.")

    # 5. Create our project sample (10,000 AmazonHelp inbound tweets)

    # Filter for inbound tweets that mention @AmazonHelp
    df_amazonhelp_inbound = df[(df['inbound'] == True) & (df['text'].str.contains('@AmazonHelp', na=False, case=False))]

    # Determine the sample size, ensuring we don't try to sample more than available rows
    sample_size = min(10000, len(df_amazonhelp_inbound))

    if sample_size > 0:
        df_brand = df_amazonhelp_inbound.sample(n=sample_size, random_state=42)
        print(f"✅ Sample of {len(df_brand)} AmazonHelp inbound tweets created in 'df_brand'.")
        print(df_brand.head())
        # Save the sample to CSV
        output_csv = os.path.join(os.path.dirname(__file__), 'sample_amazonhelp.csv')
        df_brand.to_csv(output_csv, index=False)
        print(f"✅ Sample saved to {output_csv}")
    else:
        print("❌ Error: No AmazonHelp inbound tweets found to sample using the current criteria.")
else:
    print("❌ Error: Could not find the CSV file. Check the dataset name.")
    