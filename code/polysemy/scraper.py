import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

url='https://baraha.com/kannada/browse.php'
chrome_driver_path = "C:/Users/ADMIN/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"

# # Create a new instance of Options
chrome_options = Options()
# chrome_options.binary_location = "C:/Users/ADMIN/Downloads/chrome-win/chrome-win/chrome.exe"

# Initialize the Chrome driver
driver = webdriver.Chrome(service=Service("C:/Users/ADMIN/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"), options=chrome_options)

wait = WebDriverWait(driver, 20)  # Increase timeout to 20 seconds
# Open the webpage
driver.get(url)

# Wait until the specific element is loaded and is clickable
element = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, '//a[@onclick="return dig_into(2, 93, \'ಅ\');"]'))
)

# Click the element
element.click()

# Get the page source after clicking
html = driver.page_source

# Parse the HTML with BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

# Now you can use BeautifulSoup to find the results you need
# results = soup.find_all('your_result_element')

print(soup.prettify())

# Don't forget to close the driver
driver.quit()


