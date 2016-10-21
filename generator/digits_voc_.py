__author__ = 'MA573RWARR10R'


def get_character_voc():
    def get_chars(start_char, end_char):
        return [chr(i) for i in range(ord(start_char), ord(end_char) + 1)]

    all_chars = get_chars('0', '9')
    # RETURNS VOCABULARY, CHARS
    return {char: i for i, char in enumerate(all_chars)}, all_chars

if __name__ == "__main__":
    print get_character_voc()
