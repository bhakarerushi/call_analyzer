import multiprocessing
import time


def generate_call_summary(call_log_object):
    """
    pip install transformers
    pip install torch
    pip install tensorflow


    """

    from transformers import BartForConditionalGeneration, BartTokenizer
    # from transformers import pipeline

    # Load pre-trained model and tokenizer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    # Input text
    # text = """
    # We begin our analysis on the neighborhood by considering the degree (k) and
    # the strength (s) distributions. Degree and strength distributions give informa-
    # tion about the level of interaction of a mobile user on the basis of the number of
    # people contacting him/her, the number of people s/he contacts and how often.
    # 295
    #
    # """
    # Preprocess the text
    input_ids = tokenizer.encode(call_log_object.call_transcript, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    call_log_object.call_summary = summary


def test_func(arg1, arg2):
    time.sleep(2)  # Simulate a time-consuming task
    result = arg1 + arg2
    print(f"Result: {result}")


def create_process(target_func, **kwargs):
    process = multiprocessing.Process(target=target_func, kwargs=kwargs)
    process.start()
    # No join here, so the main process won't wait for the subprocess to finish


if __name__ == "__main__":
    create_process(generate_call_summary, arg1=10, arg2=20)
    print("Main process continues immediately...")
