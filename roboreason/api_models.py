import os
import json
import logging
import uuid
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

import cv2
import io
import base64



def image_to_base64(img):
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img = img
    pil_img = Image.fromarray(rgb_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


user_prompt_template_gemini = (
    "Here are images containing multiple camera views of a robot attempting to complete a task. "
    + "The first image is from the previous timestep. The second image is from the current timestep. "
    + "The task description is: {task_description}. "
    + "The predicted task progress for the previous timestep was {prev_progress}%. Predict the task progress for the current timestep. "
    + "Note that the previous progress value might not be accurate. Please carefully assess the images to determine the correct progress for the current timestep. "
    + "Also note that the performance of the robot is unknown so progress can increase or decrease at any timestep. "
    + "Before providing your final answer, first briefly provide a few words that reason about what is happening at the current timestep relative to the previous timestep. "
    + "Your reasoning process should be no more than one or two sentences and MUST BE enclosed within <think> </think> tags. IMPORTANT: Do not refer to 'the user' in your reasoning. Only reason as though it is an internal monologue thinking about the robot. If you are unsure about the progress, please try to avoid overestimation and provide a conservative estimate. Please refer to the images as timesteps instead of as separate images. "
    + "Your final answer MUST BE enclosed within <answer> </answer> tags and should be an integer representing current task progress percentage. "
    + "Example output format: <think>[detailed reasoning process]</think><answer>[current task progress]%</answer>"
)

user_prompt_template_gpt = (
    "Here are images containing multiple camera views of a robot attempting to complete a task. "
    + "The first image is from the previous timestep. The second image is from the current timestep. "
    + "The task description is: {task_description}. "
    + "The predicted task progress for the previous timestep was {prev_progress}%. Predict the task progress for the current timestep. "
    + "Before providing your final answer, first briefly provide a few sentences that reason about what is happening at the current timestep relative to the previous timestep. "
    + "Your reasoning process should be no more than a few sentences and MUST BE enclosed within <think> </think> tags. Please refer to the images as timesteps instead of as separate images. "
    + "Your final answer MUST BE enclosed within <answer> </answer> tags and should be an integer representing current task progress percentage. "
    + "Example output format: <think>[detailed reasoning process]</think><answer>[current task progress]%</answer>"
)

replacements = {
    '\u2018': "'",  # left single quote
    '\u2019': "'",  # right single quote
    '\u201c': '"',  # left double quote
    '\u201d': '"',  # right double quote
    '\u2013': '-',  # en dash
    '\u2014': '-',  # em dash
}


def gemini(client, model, task_description_i, frame_list, try_count_max=3, verbose=False):
    from google.genai import types
    # 
    # image_file_num_list_idx = list(range(1,len(frame_list)))
    # 
    current_progress = 0
    prompt_list = []
    progress_list = []
    response_text_list = []
    messages_content = []
    response_text = None
    for current_idx in range(1,len(frame_list)):
        try_count=0
        while try_count < try_count_max:
            # if True:
            try:
                if True:
                    # messages_content = [
                    #     user_prompt_template_gemini.format(task_description=task_description_i, prev_progress=current_progress),
                    #     {'mime_type':'image/jpeg', 'data': frame_list[image_file_num_list_idx[current_idx]-1]},
                    #     {'mime_type':'image/jpeg', 'data': frame_list[image_file_num_list_idx[current_idx]]}
                    # ]
                    # response = client.generate_content(messages_content, generation_config=genai.GenerationConfig(max_output_tokens=6000,temperature=0,top_k=1))
                    # response = response.to_dict()
                    # response_text = response['candidates'][0]['content']['parts'][0]['text']
                    image_prev_bytes = base64.b64decode(frame_list[current_idx-1])
                    image_current_bytes = base64.b64decode(frame_list[current_idx])
                    messages_content=[
                            user_prompt_template_gemini.format(task_description=task_description_i, prev_progress=current_progress),
                            types.Part.from_bytes(
                                data=image_prev_bytes,
                                mime_type='image/jpeg',
                            ),
                            types.Part.from_bytes(
                                data=image_current_bytes,
                                mime_type='image/jpeg',
                            )
                        ]
                    response = client.models.generate_content(
                        model=model,
                        contents=messages_content
                    )
                    response_text = response.text
                # 
                for src, target in replacements.items():
                    response_text = response_text.replace(src, target)
                # 
                current_progress = int(response_text.split('<answer>')[1].split('</answer>')[0].replace('%','').strip())
                progress_list.append(current_progress)
                response_text_list.append(response_text)
                prompt_list.append(messages_content[0])
                break
            except Exception as e:
                if verbose:
                    print(f"\nError: {e}")
                try_count += 1
                if 'quota' in str(e).lower():
                    import time
                    time.sleep(1*60)
                if verbose:
                    print(f'Response: {response_text}')
        # 
        if try_count >= try_count_max:
            response_text_list.append('')
            if len(progress_list)>0:
                current_progress = progress_list[-1]
            else:
                current_progress = 0
            progress_list.append(current_progress)
            if len(messages_content)>0:
                prompt_list.append(messages_content[0])
            else:
                prompt_list.append('')
        # 
        if verbose:
            # print('\n\n*******************************************************************************')
            # print(prompt_list[-1])
            print(f'\n----------------- Timestep {current_idx} -----------------')
            print(response_text_list[-1])
            print(progress_list[-1])
    # 
    return progress_list, response_text_list, prompt_list




def gpt(client, model, task_description_i, frame_list, try_count_max=3, verbose=False):
    # image_file_num_list_idx = list(range(1,len(frame_list)))
    # 
    current_progress = 0
    prompt_list = []
    progress_list = []
    response_text_list = []
    messages_content = []
    response_text = None
    for current_idx in range(1,len(frame_list)):
        try_count=0
        while try_count < try_count_max:
            # if True:
            try:
                if True:
                    # base64_image_prev = frame_list[image_file_num_list_idx[current_idx]-1]
                    # base64_image_current = frame_list[image_file_num_list_idx[current_idx]]
                    base64_image_prev = frame_list[current_idx-1]
                    base64_image_current = frame_list[current_idx]
                    messages_content = [
                        {"type": "text", "text": user_prompt_template_gpt.format(task_description=task_description_i, prev_progress=current_progress)},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_prev}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}"}}
                    ]
                    # 
                    response = client.chat.completions.create(model=model,messages=[{"role": "user","content": messages_content}])
                    response_text=response.choices[0].message.content
                # 
                for src, target in replacements.items():
                    response_text = response_text.replace(src, target)
                # 
                current_progress = int(response_text.split('<answer>')[1].split('</answer>')[0].replace('%','').strip())
                progress_list.append(current_progress)
                response_text_list.append(response_text)
                prompt_list.append(messages_content[0])
                break
            except Exception as e:
                if verbose:
                    print(f"\nError: {e}")
                try_count += 1
                if 'quota' in str(e).lower():
                    import time
                    time.sleep(1*60)
                if verbose:
                    print(f'Response: {response_text}')
        # 
        if try_count >= try_count_max:
            response_text_list.append('')
            if len(progress_list)>0:
                current_progress = progress_list[-1]
            else:
                current_progress = 0
            progress_list.append(current_progress)
            if len(messages_content)>0:
                prompt_list.append(messages_content[0])
            else:
                prompt_list.append('')
        # 
        if verbose:
            # print('\n\n*******************************************************************************')
            # print(prompt_list[-1])
            # print('\n----------------- Response -----------------')
            print(f'\n----------------- Timestep {current_idx} -----------------')
            print(response_text_list[-1])
            print(progress_list[-1])
    # 
    return progress_list, response_text_list, prompt_list




def api_models(model, video, task_description, key, view_type_per_video=None, context_window = ['current', 'previous', 'first'], verbose=False):
    # 
    if 'gpt' in model.lower():
        from openai import OpenAI
        client = OpenAI(api_key=key)
    elif 'gemini' in model.lower():
        from google import genai
        client = genai.Client(api_key=key)
        # response = client.models.generate_content(model="gemini-3-flash-preview", contents="Explain how AI works in a few words")
        # response = client.models.generate_content(model="gemini-3-pro-preview", contents="Explain how AI works in a few words")
        # print(response.text)
        # import google.generativeai as genai
        # genai.configure(api_key=key)
        # client = genai.GenerativeModel(model_name = model, system_instruction = "You are an expert roboticist with the goal of predicting task progress percentages given frames from a video of a robot attempting to complete a task.")
    else:
        raise ValueError(f"Unknown model name: {model}")
    # 
    video_base64 = [image_to_base64(frame) for frame in video]
    # 
    if 'gpt' in model.lower():
        progress_list, response_text_list, prompt_list = gpt(client, model.lower(), task_description, video_base64, verbose=verbose)
    else:
        progress_list, response_text_list, prompt_list = gemini(client, model.lower(), task_description, video_base64, verbose=verbose)
    # 
    # assuming no reasoning or progress prediction at first timestep
    response_text_list = [''] + response_text_list
    # 
    valid_answer_list = [0]  
    for ans in progress_list:
        try:
            progress = int(ans)
        except (ValueError, TypeError):
            progress = valid_answer_list[-1]
        valid_answer_list.append(progress)
    # 
    return valid_answer_list, response_text_list





