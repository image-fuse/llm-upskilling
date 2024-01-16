from bs4 import BeautifulSoup
import requests
import json
import os


def scrape_and_save_quotes(url, output_folder, output_filename):
    """
    Scrapes quotes data from a website and saves it in a JSON file.

    Args:
    -----
        url (str): The base URL of the website to scrape quotes from.
        output_folder (str): The folder where the JSON file will be saved.
        output_filename (str): The name of the output JSON file.

    Returns:
    -------
        None
    """
    base_url = url
    page_number = 1
    prev_url = None

    quotes_data = []  # Create a list to store quotes data

    while url != prev_url:
        html_text = requests.get(url).text
        soup = BeautifulSoup(html_text, 'lxml')

        quotes = soup.find_all('div', class_='quote')

        for quote in quotes:
            quote_text = quote.find('span', class_='text').text
            quote_author = quote.find('small', itemprop='author').text
            quote_tags = quote.find_all('a', class_='tag')
            quote_tags_list = [tag.text for tag in quote_tags]

            # print(quote_text)
            # print(quote_author)
            # print(quote_tags_list)
            # print()

            # Create a dictionary to store quote data
            quote_data = {
                "Quote Text": quote_text,
                "Author": quote_author,
                "Tags": quote_tags_list
            }

            quotes_data.append(quote_data)

        print(f"Scraped page {page_number}")

        prev_url = url  # Store the current URL
        next_button = soup.find('li', class_='next')

        if next_button:
            page_number += 1
            url = f'{base_url}/page/{page_number}/'
        else:
            break

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the scraped data to a JSON file in the specified output folder
    json_file_path = os.path.join(output_folder, output_filename)
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(quotes_data, json_file, ensure_ascii=False, indent=4)

    print("Data saved to", json_file_path)


if __name__ == "__main__":
    # Define the base URL of the website to scrape, the output folder,
    # and the output JSON file name
    website_url = 'https://quotes.toscrape.com'
    output_directory = 'data'  # Output folder name
    output_filename = 'quotes_data.json'  # Output JSON file name

    # Call the function to scrape and save quotes data
    scrape_and_save_quotes(website_url, output_directory, output_filename)
