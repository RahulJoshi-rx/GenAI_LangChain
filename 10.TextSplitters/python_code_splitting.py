from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

    def have_birthday(self):
        self.age += 1
        print(f"Happy Birthday, {self.name}! You are now {self.age} years old.")


def main():
    # Create an object of the Person class
    person = Person("Rahul", 30)
    person.greet()
    person.have_birthday()


# Standard boilerplate to call main()
if __name__ == "__main__":
    main()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 400,
    chunk_overlap=0
)

chunks = splitter.split_text(text=text)

print(len(chunks))
print('************************************************************')
print(chunks[0])
print('************************************************************')
print(chunks[1])
