import os

with open('D:/sd/Practices/any2md/output/R/mtswr/chunk_temp.md', 'r', encoding='utf-8') as f:
    content = f.read()

with open('D:/sd/Practices/any2md/output/R/mtswr/24-15_ko.md', 'a', encoding='utf-8') as f:
    f.write('\n' + content)

os.remove('D:/sd/Practices/any2md/output/R/mtswr/chunk_temp.md')
