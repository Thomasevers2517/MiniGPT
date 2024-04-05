"""
This code was used to add a special token preceding each speaker in 
the Shakespare dataset. 
"""
def get_speakers(text: str) -> int:
    lines = text.split('\n')
    speakers = set()
    for i, line in enumerate(lines):
        # new speaker.
        if line.endswith(':') and (i == 0 or lines[i-1] == ''):
            speakers.add(line)
    return list(speakers)

def generate_speaker_tokens(speakers: set, start_idx: int) -> dict:
    tokens_to_ids = {}
    speakers_to_tokens = {}
    for i, speaker in enumerate(speakers):
        token_id = start_idx + i
        token_name = f"<|speaker{i}|>"
        tokens_to_ids[token_name] = token_id
        speakers_to_tokens[speaker] = token_name
    return tokens_to_ids, speakers_to_tokens

def split_dialogue(text: str, speakers_to_tokens: dict) -> str:
    lines = text.split('\n')
    dialogues = ""
    speaker = None
    dialogue = ""
    for i, line in enumerate(lines):
        # new speaker.
        if line.endswith(':') and (i == 0 or lines[i-1] == ''):
            if speaker:
                # Store previous speaker's dialogue.
                delimiter = speakers_to_tokens[speaker]
                dialogues += f"{delimiter}\n{speaker}\n{dialogue}"
            # Update current speaker
            speaker = line
            # Reset dialogue for the new speaker
            dialogue = ""  
        else: 
            dialogue += f"{line}\n"
        
    # Append the last speaker's dialogue
    if speaker:
        delimiter = speakers_to_tokens[speaker]
        dialogues += f"{delimiter}\n{speaker}\n{dialogue}"
    
    return dialogues



# CODE TO RUN SPECIFIC TOKENS
# add special tokens for speakers.
# speakers = get_speakers(text)
# tokens_to_ids, speakers_to_tokens = generate_speaker_tokens(speakers, tokenizer.vocab_size)
# # extend tokenizer with new special tokens.
# tokenizer.extend(tokens_to_ids)
# # split text into dialogues with speaker tokens.
# dialogues = split_dialogue(text, speakers_to_tokens)
# # encode text.
# tokens = tokenizer.encode(dialogues)