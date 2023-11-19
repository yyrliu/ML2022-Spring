import opencc

def main(input_file, output_file):
    converter = opencc.OpenCC('s2twp.json')

    with open(output_file, 'w', encoding="utf-8") as f_out:
        with open(input_file, 'r', encoding="utf-8") as f_in:
            for line in f_in:
                converted_line = converter.convert(line)
                f_out.write(converted_line)

if __name__ == '__main__':
    main('data/downloaded/test/TED2013.en-zh.zh', 'data/processed/test.raw.zh')
