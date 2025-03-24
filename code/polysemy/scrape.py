from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd

url='https://baraha.com/kannada/browse.php'
chrome_driver_path = "C:/Users/ADMIN/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"

driver = webdriver.Chrome(service=Service(chrome_driver_path))

# Navigate to your webpage
driver.get(url)

# Find the table with cellpadding="20"
table = driver.find_element(By.CSS_SELECTOR, 'table[cellpadding="20"]')

# Find the first link within the table and click on it
link = table.find_element(By.TAG_NAME, 'a')
link.click()

# Find the table with cellpadding="20"
table = driver.find_element(By.CSS_SELECTOR, 'table[cellpadding="20"]')

# Find the first link within the table and click on it
link = table.find_element(By.TAG_NAME, 'a')
link.click()

data = []

# Loop over the pages until there are 20580 entries in the DataFrame
while len(data) < 20580:
    # Wait for the new page to load
    driver.implicitly_wait(10)  # wait up to 10 seconds for the new page to load

    # Get the page source
    html_doc = driver.page_source

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_doc, 'html.parser')

    for p in soup.find_all('p'):
        word = p.find('span', class_='word')
        wordtype = p.find('span', class_='wordtype')
        meaning = p.find_next_sibling('ul')

        if word and wordtype and meaning:
            data.append({
                'word': word.text,
                'wordtype': wordtype.text,
                'meaning': meaning.text.strip()
            })

    # Click on the "next page" link if there are less than 20580 entries
    if len(data) < 20580:
        next_page_link = driver.find_element(By.XPATH, '//td[@align="right"]/a')
        next_page_link.click()

# Store the data in a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('data.csv', index=False)