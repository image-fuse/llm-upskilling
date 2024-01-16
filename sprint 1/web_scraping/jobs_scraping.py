from bs4 import BeautifulSoup
import requests
import json
import os


def scrape_and_save_job_data(url, output_folder):
    """
    Scrapes job data from a website and saves it in a JSON file.

    Args:
    -----
        url (str): The URL of the website to scrape job data from.
        output_folder (str): The folder where the JSON file will be saved.

    Returns:
    --------
        None
    """
    # Send an HTTP GET request to the URL and retrieve the HTML content
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'lxml')
    jobs = soup.find_all('div', class_='card-content')

    job_data_list = []  # Create a list to store job data

    for job in jobs:
        # Extract job details from the HTML elements
        job_title = job.find('h2', class_='title is-5').text.strip()
        company_name = job.find(
            'h3', class_='subtitle is-6 company').text.strip()
        location = job.find('p', class_='location').text.strip()
        published_date = job.find('time').text.strip()
        link = job.find('a', class_='card-footer-item', text='Apply')['href']

        # print("Job Title:", job_title.strip())
        # print("Company:", company_name.strip())
        # print("Location:", location.strip())
        # print("Published Date:", published_date.strip())
        # print("Apply Link:", link)
        # print()

        # Create a dictionary to store job data
        job_data = {
            "Job Title": job_title,
            "Company": company_name,
            "Location": location,
            "Published Date": published_date,
            "Apply Link": link
        }

        job_data_list.append(job_data)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the scraped data to a JSON file in the specified output folder
    json_file_path = os.path.join(output_folder, 'job_data.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(job_data_list, json_file, indent=4)

    print("Data saved to", json_file_path)


if __name__ == "__main__":
    # Define the URL of the website to scrape and the output folder
    website_url = 'https://realpython.github.io/fake-jobs'
    output_directory = 'data'  # Output folder name

    # Call the function to scrape and save job data
    scrape_and_save_job_data(website_url, output_directory)
