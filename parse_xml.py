import os
import xml.etree.ElementTree as ET

e_books = 0
not_e_books = 0

for file in os.listdir("./books_xml"):
    tree = ET.parse("./books_xml/" + file)
    root = tree.getroot()

    book = root.find("book")

    for child in book:
        if child.tag == "is_ebook":
            if child.text == "true":
                e_books += 1
            elif child.text == "false":
                not_e_books += 1


print(f"Number of ebooks: {e_books}")
print(f"Number of non-ebooks: {not_e_books}")
