
def process_slimorca(examples):
    # assume single example
    try:
        system, human, gpt = examples['conversations']
        sanity_check = system['from'] == 'system' and human['from'] == 'human' and gpt['from'] == 'gpt'

        reformat_txt = f"### System : {system['value']}"
        examples['text'] = reformat_txt if sanity_check else ""
    except:
        reformat_txt = ""
        examples['text'] = reformat_txt

    return examples