import numpy as np
import math

class Book:
    def __init__(self,title,author,isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_borrowed = False
        
    def  borrow_book(self):
        self.is_borrowed  = True
        
    def  return_book(self):
        self.is_borrowed = False

class Member:
    def __init__(self,name,member_id):
        self.name = name
        self.member_id = member_id
        self.borrowed_books = []
    
    def borrow(self,book):
        self.borrowed_books.append(book)
        book.is_borrowed  = True
        
    def return_book(self,book):
        self.borrowed_books.remove(book)
        book.is_borrowed = False

class Library:
    def __init__(self,name):
        self.name = name
        self.books = []
        self.members = []
    
    def add_book(self,book):
        self.books.append(book)
    
    def remove_book(self,book):
        self.books.remove(book)
    
    def add_member(self,member):
        self.members.append(member)
    
    def remove_member(self,member):
        self.members.remove(member)
        
    def list_books(self):
        for book in self.books:
            status = "Borrowed" if book.is_borrowed else "Available"
            print(f"{book.title} by {book.author} (ISBN: {book.isbn}) - {status}")

    def list_available_books(self):
        available_books = [book for book in self.books if not book.is_borrowed]
        if available_books:
            for book in available_books:
                print(f"{book.title} by {book.author} (ISBN: {book.isbn}) - Available")
        else:
            print("No books are currently available.")

my_library = Library("Community Library")

book1 = Book("The Great Gatsby","F. Scott Fitzergerald","123456789")
book2 = Book("1984","George Orwell","987654321")
my_library.add_book(book1)
my_library.add_book(book2)

member1 = Member("Alice","M001")
member2 = Member("Bob","M002")
my_library.add_member(member1)
my_library.add_member(member2)

member1.borrow(book1)

my_library.list_available_books()

member1.return_book(book1)

my_library.list_books()
