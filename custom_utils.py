def print_on_file(text, filename='results.txt'):
    file = open(filename, 'a+')
    print(text, file=file)
    file.close()
