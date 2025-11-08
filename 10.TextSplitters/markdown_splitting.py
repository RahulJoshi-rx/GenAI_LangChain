from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
# Sample Markdown Document

## 1. Introduction
This is a **sample Markdown** document.  
Markdown is a lightweight markup language that allows you to format text easily.  
You can write plain text, and it will render nicely in most editors or on websites like GitHub.

---

## 2. Text Formatting

**Bold text**  
*Italic text*  
***Bold and italic text***  
~~Strikethrough text~~

> “This is a blockquote — useful for highlighting quotes or key information.”

---

## 3. Lists

### Unordered List:
- Apples  
- Bananas  
- Cherries  

### Ordered List:
1. Step one  
2. Step two  
3. Step three  

### Nested List:
- Programming Languages
  - Python
  - JavaScript
  - C++

---

## 4. Code Examples

### Inline Code
Use backticks for inline code: `print("Hello, World!")`

### Code Block
```python
def greet(name):
    print(f"Hello, {name}!")

greet("Rahul")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 200,
    chunk_overlap=0
)

chunks = splitter.split_text(text=text)

print(len(chunks))
print('************************************************************')
print(chunks[0])
print('************************************************************')
print(chunks[1])
