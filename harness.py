# here I will import the main module from your code - you need to make sure your code imports without a problem
# As per the assignment specification, your main module must be called svhn.py
import svhn


def main():
    # I might start by calling on your code to do some processing based on the model that you already trained
    result1 = svhn.test("1.png")
    print(result1)
    # i might also test with a PNG
    result2 = svhn.test("2.jpg")
    print(result2)

    # I will also call to start training on your code from scratch. I might not always wait for training to complete
    # but I will start the training and make sure it is progressing.
    average_f1_scores = svhn.traintest()
    print(average_f1_scores)


if __name__ == '__main__':
    main()
