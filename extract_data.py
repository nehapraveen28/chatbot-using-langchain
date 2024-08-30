import os
from langchain.loaders import URLLoader
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Step 1.3: Set up the URL loader
url = "https://brainlox.com/courses/category/technical"
url_loader = URLLoader(url)

# Step 1.4: Extract the data
data = url_loader.load()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(data, 'html.parser')

# Extract course data
courses = soup.find_all('div', {'class': 'course-card'})
course_data = []
for course in courses:
    title = course.find('h2', {'class': 'course-title'}).text.strip()
    description = course.find('p', {'class': 'course-description'}).text.strip()
    course_data.append({'title': title, 'description': description})

# Print the extracted course data
for course in course_data:
    print(f"Title: {course['title']}")
    print(f"Description: {course['description']}")
    print('---')
