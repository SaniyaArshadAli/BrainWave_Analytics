import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import mne
import pickle5 as pickle
from sklearn.preprocessing import LabelEncoder 
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Load pre-trained model and tokenizer
model_name = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Predefined responses
healthcare_responses = {
    "hello": "Hello! How can I assist you today?",
    "what is your name": "I am the AI bot. How can I help you?",
    "goodbye": "Goodbye! Take care.",
    "eeg wave": [
        "Electroencephalography (eeg) is a technique used to measure brain activity.",
        "eeg measures electrical signals produced by the brain's neurons.",
        "eeg waves can be classified into different frequency bands like delta, theta, alpha, beta, and gamma.",
        "Delta waves are associated with deep sleep and relaxation.",
        "Theta waves are often observed during meditation and light sleep.",
        "Alpha waves are dominant during a relaxed and awake state.",
        "Beta waves are associated with active thinking and concentration.",
        "Gamma waves are involved in information processing and cognitive tasks.",
    ],
    "mental disorders": [
        "Some common mental disorders include depression, anxiety, schizophrenia, and bipolar disorder.",
        "Depression is characterized by persistent feelings of sadness and loss of interest.",
        "Anxiety disorders involve excessive worry and fear.",
        "Schizophrenia is a severe mental disorder that affects thinking, emotions, and behavior.",
        "Bipolar disorder involves mood swings between depression and mania.",
    ],
    "tell me a joke": ["Why don't scientists trust atoms? Because they make up everything!", "Why did the scarecrow win an award? Because he was outstanding in his field!"],
    "thank you": ["You're welcome!", "No problem!", "Anytime!"],
    "who created you": ["I was created by OpenAI.", "My creators are brilliant developers.", "The masterminds behind me are talented engineers at OpenAI."],
    "what is the meaning of life": ["That's a deep philosophical question. There are many theories, but I'm just a bot!", "The meaning of life is subjective and varies for each individual."],
    "how old are you": ["I am a virtual AI bot, so I don't have an age.", "I don't age like humans. I'm always up-to-date!"],
    "what can you do": ["I can answer your questions,tell you about disorders and have a friendly conversation with you!", "I'm designed to assist you with information and provide a helpful chat experience."],
    "tell me about yourself": ["I am an AI chatbot designed to engage in conversations and provide helpful information.", "I'm here to make your life easier by answering your queries!"],
    "are you real": ["I'm an AI program, so I'm not real like a human. But I'm here to help!", "I'm as real as the digital world allows me to be!"],
    "do you dream": ["I don't dream, but I'm always ready to chat with you!", "Dreaming is not within my capabilities, but I'm here to assist you anytime."],
    "where are you from": ["I exist in the digital world, so I'm everywhere and nowhere at the same time!", "I'm from the servers of OpenAI, available to help you wherever you are."],
    "how can i contact you": ["You can reach out to my creators at OpenAI for any inquiries.", "I'm a virtual assistant, but you can find more about my developers at OpenAI's website."],
    "what is eeg?": "eeg stands for Electroencephalography. It is a test that measures electrical activity in the brain using electrodes placed on the scalp.",
    "What are eeg waves?": "eeg waves are the different patterns of electrical activity recorded by eeg. They include alpha, beta, theta, delta, and gamma waves.",
    "What can eeg be used for?": "EEG can be used to diagnose and monitor various neurological conditions such as epilepsy, sleep disorders, and brain injuries.",
    "How is eeg performed?": "During an eeg, electrodes are attached to the scalp, and the brain's electrical activity is recorded while the patient is at rest or performing specific tasks.",
    "What is the significance of alpha waves in eeg?": "Alpha waves are associated with a relaxed and awake state. Their presence can indicate a state of calmness and relaxation.",
    "What do beta waves indicate in EEG?": "Beta waves are usually seen when a person is alert and actively thinking. They are associated with cognitive and mental activities.",
    "What are theta waves in eeg?": "Theta waves are commonly observed during light sleep or in a meditative state. They may also be present in certain neurological conditions.",
    "What is the role of delta waves in eeg?": "Delta waves are the slowest brainwaves and are seen during deep sleep. They play a crucial role in the restorative process of sleep.",
    "Can eeg detect seizures?": "Yes, EEG is one of the primary tools for diagnosing and monitoring seizures. It helps identify abnormal electrical activity in the brain.",
    "Are there any risks associated with eeg?": "EEG is a safe and non-invasive procedure. There are no significant risks involved, and it does not cause any pain.",
    "Can eeg be used for diagnosing psychiatric disorders?": "eeg can provide valuable insights into certain psychiatric disorders, but it is not used as the sole diagnostic tool for most mental health conditions.",
    "How long does an eeg test take?": "An eeg test usually takes around 30 minutes to an hour, depending on the specific testing requirements.",
    "What is quantitative eeg (qeeg)?": "Quantitative eeg (qeeg) is a method that quantifies and analyzes EEG data to provide more detailed information about brain function.",
    "Is eeg used in brain-computer interfaces?": "Yes, eeg is used in brain-computer interfaces to allow individuals to control external devices using their brain activity.",
    "Can eeg be used to assess cognitive function?": "Yes, eeg can be used to assess cognitive function and study brain responses to different cognitive tasks.",
    "What is the role of eeg in sleep studies?": "eeg is commonly used in sleep studies to monitor brain activity during different sleep stages and identify sleep disorders.",
    "Can eeg be used to determine brain death?": "eeg is one of the tools used to assess brain function in patients with severe brain injury or in determining brain death.",
    "What is the difference between eeg and fMRI?": "eeg measures electrical brain activity, while fMRI (functional Magnetic Resonance Imaging) measures changes in blood flow in the brain.",
    "Can eeg be used to predict neurological disorders?": "EEG patterns can sometimes provide early indications of neurological disorders, but further tests and evaluations are needed for a definitive diagnosis.",
    "How is eeg data analyzed?": "eeg data is analyzed by experts who look for specific patterns and abnormalities in the brain's electrical activity.",
    "What is the future of eeg technology?": "The future of eeg technology is promising, with advancements in signal processing, brain-computer interfaces, and applications in various fields.",
    "Is eeg used in research on consciousness?": "Yes, eeg is frequently used in studies related to consciousness, brain-computer interfaces, and understanding the neural basis of awareness.",
    "Can eeg be used to detect brain tumors?": "eeg is not typically used as a primary tool for detecting brain tumors. Other imaging techniques like MRI or CT scans are more commonly used.",
    "How is eeg used in neurofeedback therapy?": "In neurofeedback therapy, EEG is used to monitor brain activity, and patients are given real-time feedback to learn to self-regulate their brainwaves.",
    "Can eeg detect Alzheimer's disease?": "eeg can provide some insights into brain function changes associated with Alzheimer's disease, but it is not a definitive diagnostic tool for the condition.",
    "What is the role of eeg in anesthesia monitoring?": "eeg can be used during anesthesia to monitor the patient's brain activity and ensure the appropriate depth of anesthesia.",
    "Are there any limitations of eeg?": "Yes, eeg has some limitations, such as its inability to precisely locate the source of brain activity and sensitivity to movement artifacts.",
    "How can eeg be used in brain-computer gaming?": "eeg can be used in brain-computer gaming to allow players to control the game using their brain activity.",
    "Can eeg be used in biofeedback therapy?": "Yes, eeg biofeedback therapy uses eeg to teach individuals how to regulate their brainwaves for various therapeutic purposes.",
    "alpha waves": [
        "Alpha waves are brain waves with a frequency range of 8 to 13 Hz. They are often associated with relaxation and a wakeful but resting state.",
        "Alpha waves are dominant when you close your eyes and relax. They are an important indicator of a calm and relaxed mind.",
    ],
    "beta waves": [
        "Beta waves have a frequency range of 13 to 30 Hz. They are associated with active thinking, problem-solving, and alertness.",
        "Beta waves are commonly observed when you are awake and engaged in mental activities, such as studying or working.",
    ],
    "theta waves": [
        "Theta waves have a frequency range of 4 to 7 Hz. They are often linked to deep relaxation, meditation, and the early stages of sleep.",
        "Theta waves are known to be present during daydreaming and in states of deep relaxation.",
    ],
    "delta waves": [
        "Delta waves have the lowest frequency range, typically between 0.5 and 4 Hz. They are associated with deep sleep and unconsciousness.",
        "Delta waves are crucial for restorative sleep and play a significant role in the body's healing and recovery processes.",
    ],
    "what are eeg waves?": "eeg waves are electrical patterns produced by the brain. They are used to study brain activity.",
    "how are eeg waves measured?": "eeg waves are measured using electrodes placed on the scalp. The electrical signals are then recorded.",
    "what do eeg waves indicate?": "eeg waves can indicate brain activity, sleep stages, and certain neurological disorders.",
    "what are the different types of eeg waves?": "The main types of eeg waves are alpha, beta, delta, theta, and gamma waves.",
    "what is the frequency range of alpha waves?": "Alpha waves have a frequency range of 8-13 Hz and are associated with relaxation and calmness.",
    "what is the frequency range of beta waves?": "Beta waves have a frequency range of 13-30 Hz and are associated with alertness and focus.",
    "what is the frequency range of delta waves?": "Delta waves have a frequency range of 0.5-4 Hz and are associated with deep sleep.",
    "what is the frequency range of theta waves?": "Theta waves have a frequency range of 4-8 Hz and are associated with daydreaming and creativity.",
    "what is the frequency range of gamma waves?": "Gamma waves have a frequency range of 30-100 Hz and are associated with cognitive processing.",
    "what factors can affect eeg wave patterns?": "Several factors can affect eeg wave patterns, including sleep, medication, and neurological conditions.",
    "how is eeg used in diagnosing epilepsy?": "eeg is commonly used to diagnose epilepsy by detecting abnormal electrical activity in the brain.",
    "what is an eeg sleep study?": "An eeg sleep study is used to monitor brain activity during sleep to assess sleep disorders.",
    "what is brain mapping using eeg?": "Brain mapping with eeg is a technique to identify brain regions responsible for specific functions.",
    "can eeg be used to diagnose mental disorders?": "eeg can provide valuable information in diagnosing certain mental disorders.",
    "how can eeg help in cognitive research?": "eeg is used in cognitive research to study brain processes related to attention, memory, and decision-making.",
    "what are event-related potentials (ERPs) in eeg?": "ERPs are brain responses recorded with eeg in response to specific stimuli.",
    "what are artifacts in eeg recordings?": "Artifacts are unwanted signals in eeg recordings caused by external interference or body movements.",
    "how is eeg used in the study of brain-computer interfaces?": "eeg is used in brain-computer interfaces to translate brain signals into commands for external devices.",
    "what are the limitations of eeg as a brain imaging technique?": "eeg has limitations in spatial resolution compared to other brain imaging techniques like fMRI.",
    "what are the ethical considerations in eeg research?": "Ethical considerations in eeg research include informed consent and participant confidentiality.",
    "how can eeg be used in monitoring anesthesia during surgery?": "eeg can be used to monitor the depth of anesthesia during surgery to ensure patient safety.",
    "what are the implications of eeg in studying sleep disorders?": "eeg helps in understanding sleep disorders and developing effective treatments.",
    "how can eeg help in diagnosing neurological conditions?": "eeg can help diagnose neurological conditions such as epilepsy, stroke, and brain tumors.",
    "what are the challenges in interpreting eeg data?": "Interpreting eeg data can be challenging due to variability and complexity of brain signals.",
    "what are slow waves in eeg recordings?": "Slow waves are low-frequency eeg waves associated with deep sleep and certain brain functions.",
    "what is brainwave entrainment using eeg?": "Brainwave entrainment uses external stimuli to synchronize brainwaves with desired frequencies.",
    "how can eeg be used in neurofeedback therapy?": "eeg neurofeedback therapy helps individuals learn to self-regulate their brain activity.",
    "what are sleep spindles and K-complexes in eeg sleep studies?": "Sleep spindles and K-complexes are characteristic waveforms in eeg during sleep.",
    "what are the applications of eeg in brain research?": "eeg is widely used in brain research for studying cognition, emotions, and brain disorders.",
    "what are epileptiform discharges in eeg?": "Epileptiform discharges are abnormal patterns seen in eeg associated with epilepsy.",
    "how can eeg be used in assessing brain injury and coma?": "EEG can provide information about brain function in patients with brain injury and coma.",
    "acute stress disorder": [
        "Acute Stress Disorder can occur after a traumatic event.",
        "It's essential to seek support from professionals if you experience Acute Stress Disorder.",
        "Symptoms of Acute Stress Disorder may include intrusive thoughts and avoidance behaviors.",
    ],
    "adjustment disorder": [
        "Adjustment Disorder can result from difficulties coping with life changes.",
        "If you're experiencing significant distress due to life changes, consider seeking help.",
        "Therapy and support can be beneficial for managing Adjustment Disorder.",
    ],
    "alcohol use disorder": [
        "Alcohol Use Disorder is a serious condition that can impact various aspects of life.",
        "If you or someone you know is struggling with alcohol, consider seeking help from a healthcare professional.",
        "Treatment options for Alcohol Use Disorder include therapy and support groups.",
    ],
    "bipolar disorder": [
        "Bipolar Disorder involves periods of mania and depression.",
        "It's important to work with a healthcare provider to manage Bipolar Disorder effectively.",
        "Medications and therapy can be beneficial for individuals with Bipolar Disorder.",
    ],
    "behavioral addictive disorder": [
        "Behavioral Addictive Disorder can involve addiction to activities like gambling or gaming.",
        "Seeking help from a mental health professional is essential for managing Behavioral Addictive Disorder.",
        "Cognitive-behavioral therapy may be helpful in treating Behavioral Addictive Disorder.",
    ],
    "depressive disorder": [
        "Depressive Disorder can lead to persistent feelings of sadness and loss of interest.",
        "It's essential to talk to a mental health professional if you experience symptoms of Depressive Disorder.",
        "Therapy and medications can be effective in treating Depressive Disorder.",
    ],
    "healthy control": [
        "Healthy Control refers to individuals without any diagnosed mental health disorders.",
        "Maintaining a healthy lifestyle can contribute to overall well-being in Healthy Controls.",
        "Regular exercise and self-care practices are important for Healthy Controls.",
    ],
    "obsessive compulsive disorder": [
        "Obsessive Compulsive Disorder involves intrusive thoughts and repetitive behaviors.",
        "If you have symptoms of OCD, consider consulting with a mental health specialist.",
        "Exposure and response prevention therapy are commonly used to treat OCD.",
    ],
    "panic disorder": [
        "Panic Disorder can lead to sudden and intense episodes of fear and anxiety.",
        "Seeking help from a healthcare professional is crucial if you experience Panic Disorder.",
        "Cognitive-behavioral therapy can be effective in managing Panic Disorder.",
    ],
    "post traumatic stress disorder": [
        "Post Traumatic Stress Disorder can occur after experiencing or witnessing a traumatic event.",
        "If you have symptoms of PTSD, consider reaching out to a mental health provider.",
        "Trauma-focused therapies are often used to treat Post Traumatic Stress Disorder.",
    ],
    "schizophrenia": [
        "Schizophrenia is a complex mental health condition that affects thoughts, emotions, and behavior.",
        "Early intervention and treatment can help individuals with Schizophrenia lead fulfilling lives.",
        "Antipsychotic medications and psychosocial interventions are commonly used to manage Schizophrenia.",
    ],
    "Social Anxiety Disorder": [
        "Social Anxiety Disorder involves excessive fear and anxiety in social situations.",
        "If you struggle with social anxiety, consider seeking support from a mental health professional.",
        "Cognitive-behavioral therapy is often used to treat Social Anxiety Disorder.",
    ],
    "depression": [
        "I'm really sorry to hear that you're feeling this way. Please consider talking to a mental health professional.",
        "Depression is a serious condition. Reach out to someone you trust or a mental health expert.",
        "I understand how tough it can be. Remember, you're not alone, and help is available.",
    ],
    "personality disorders": [
        "Personality disorders can be challenging, but with the right support, people can manage them effectively.",
        "Seeking professional help can be beneficial if you suspect you have a personality disorder.",
        "Learning more about personality disorders can help you understand your experiences better.",
    ],
    "anxiety disorders": [
        "Anxiety can be overwhelming, but there are coping strategies that can help you manage it.",
        "Breathing exercises and mindfulness can be helpful in managing anxiety.",
        "Consider reaching out to a therapist to work on anxiety management.",
    ],
    "schizophrenia": [
        "Schizophrenia can be difficult to manage, but with treatment, people can lead fulfilling lives.",
        "Support from family and friends can make a significant difference for someone with schizophrenia.",
        "If you suspect you have schizophrenia, consult a mental health professional as soon as possible.",
    ],
    "eating disorders": [
        "Eating disorders can be serious and require professional intervention.",
        "Seeking help from a healthcare professional experienced in treating eating disorders is essential.",
        "Remember that recovery is possible with the right support and treatment.",
    ],
    "addictive behaviors": [
        "Addictive behaviors can negatively impact your life. Consider seeking help to overcome them.",
        "Breaking free from addictive behaviors may require professional assistance and a strong support network.",
        "Recovery from addiction is possible with the right treatment and dedication.",
    ],
    "how to cope with depression": "Coping with depression involves seeking professional help, engaging in self-care activities, staying connected with supportive people, and considering therapy or medication as recommended by a healthcare professional.",
    "what causes personality disorders": "The exact cause of personality disorders is not fully understood. It is believed to be a combination of genetic, environmental, and social factors. Early intervention and therapy can help manage symptoms.",
    "how to manage anxiety": "Managing anxiety involves relaxation techniques, stress reduction, cognitive-behavioral therapy, and, in some cases, medication as prescribed by a healthcare professional.",
    "symptoms of schizophrenia": "Symptoms of schizophrenia include hallucinations, delusions, disorganized thinking, and emotional withdrawal. Early diagnosis and a comprehensive treatment plan can improve outcomes.",
    "treatment options for eating disorders": "Treatment options for eating disorders may include therapy, nutrition counseling, medical monitoring, and support groups. Early intervention is essential for a better prognosis.",
    "overcoming addictive behaviors": "Overcoming addictive behaviors requires a combination of therapy, support groups, lifestyle changes, and, in some cases, medication. Seeking help and staying committed to recovery are crucial.",
    "how to support someone with depression": "Supporting someone with depression involves being understanding, patient, and encouraging them to seek professional help. Offer your presence and listen without judgment.",
    "understanding personality disorders": "Understanding personality disorders includes recognizing the complexity of the condition and its impact on an individual's life. Empathy and support are essential when interacting with someone with a personality disorder.",
    "tips for managing anxiety": "Managing anxiety can be achieved through mindfulness, regular exercise, healthy coping mechanisms, and maintaining a balanced lifestyle.",
    "myths about schizophrenia": "There are many myths surrounding schizophrenia. It's important to educate oneself about the condition and challenge these misconceptions to reduce stigma.",
    "how to help someone with an eating disorder": "If you suspect someone has an eating disorder, express your concern gently, encourage them to seek professional help, and avoid making judgments or comments about their appearance.",
    "breaking free from addiction": "Breaking free from addiction is a challenging journey that requires determination, professional support, and a strong support network. Be kind to yourself during the process.",
    "mood disorder": "Mood disorders, such as depression and bipolar disorder, are characterized by significant changes in mood and emotions.",
    "addictive disorder": "Addictive disorders involve a dependence on substances or behaviors, leading to harmful consequences.",
    "trauma and stress related disorder": "Trauma and stress-related disorders, like PTSD, are triggered by exposure to traumatic events.",
    "schizophrenia": "Schizophrenia is a severe mental disorder characterized by distorted thinking, emotions, and perceptions.",
    "anxiety disorder": "Anxiety disorders involve excessive worry, fear, and nervousness that can interfere with daily life.",
    "healthy control": "Healthy control refers to individuals without any significant mental health conditions.",
    "obsessive compulsive disorder": "Obsessive-Compulsive Disorder (OCD) is characterized by intrusive thoughts and repetitive behaviors.",
    "substance abuse": "Substance abuse is a type of addictive disorder where individuals use drugs or alcohol in a way that leads to negative consequences on their health and life.",
    "alcoholism": "Alcoholism, also known as alcohol use disorder, is a chronic condition characterized by an inability to control alcohol consumption despite its negative impact on a person's life.",
    "drug addiction": "Drug addiction is a condition where individuals become dependent on a specific substance and find it challenging to stop using it even when it causes harm.",
    "post-traumatic stress disorder (PTSD)": "PTSD is a trauma and stress-related disorder that occurs in individuals who have experienced or witnessed a traumatic event. It can lead to flashbacks, nightmares, and emotional distress.",
    "acute stress disorder": "Acute stress disorder is a short-term condition that occurs after a traumatic event. It shares similarities with PTSD but typically lasts for a shorter duration.",
    "what are common psychiatric disorders diagnosed using EEG": "Common psychiatric disorders diagnosed using EEG include epilepsy, sleep disorders, and some cases of schizophrenia.",
    "how is EEG used in diagnosing epilepsy": "EEG is used to detect abnormal electrical activity in the brain, known as epileptiform discharges, which helps diagnose epilepsy.",
    "can EEG detect sleep disorders": "Yes, EEG is used to study sleep patterns and diagnose sleep disorders like sleep apnea, narcolepsy, and insomnia.",
    "is EEG helpful in diagnosing depression": "While EEG can provide some insights into the brain's electrical activity related to mood disorders, it is not typically used as a standalone test for diagnosing depression.",
    "what does the EEG pattern look like during a seizure": "During a seizure, EEG often shows abnormal and excessive electrical activity, reflecting the abnormal brain activity associated with the seizure.",
    "how does EEG help in diagnosing Alzheimer's disease": "EEG patterns may show characteristic changes in Alzheimer's disease, but it is not the primary diagnostic tool. Other tests, such as neuroimaging and cognitive assessments, are more commonly used for diagnosis.",
    "what is an EEG recording session like": "During an EEG recording, electrodes are placed on the scalp, and the patient is asked to lie still with their eyes closed. The EEG machine records the brain's electrical activity during the session.",
    "how long does an EEG recording session typically last": "An EEG recording session usually lasts 20 to 60 minutes, but it may be longer in some cases.",
    "what are some limitations of using EEG for psychiatric disorders": "EEG has limitations in spatial resolution and cannot pinpoint the exact location of brain abnormalities. It is often used in combination with other tests for more accurate diagnosis.",
    "can EEG be used to detect attention deficit hyperactivity disorder (ADHD)": "While EEG can provide some insights into brain activity related to attention and focus, it is not a definitive diagnostic tool for ADHD.",
    "how is EEG helpful in diagnosing brain tumors": "EEG is not primarily used for diagnosing brain tumors. Imaging techniques like MRI and CT scans are more common for detecting brain tumors.",
    "what does a normal EEG pattern look like": "A normal EEG pattern shows a balanced and rhythmic electrical activity with different frequency bands like alpha, beta, delta, and theta waves.",
    "how is EEG used in monitoring epilepsy treatment": "EEG is used to monitor brain activity in patients with epilepsy and assess the effectiveness of antiepileptic medications.",
    "what is an epileptic seizure": "An epileptic seizure is a sudden surge of abnormal electrical activity in the brain, which can lead to various physical and mental manifestations.",
    "is EEG safe": "EEG is a non-invasive and safe procedure with no known significant risks.",
    "can EEG detect migraines": "EEG is not routinely used to diagnose migraines. However, it may be used in certain cases to rule out other causes of headache.",
    "how is EEG used in diagnosing sleepwalking": "EEG can help differentiate sleepwalking from other sleep disorders by analyzing the brain's activity during sleep.",
    "is EEG used in assessing brain injuries": "Yes, EEG is used in the assessment of brain injuries to monitor brain function and detect abnormalities in the electrical activity.",
    "what is the role of EEG in studying consciousness": "EEG is valuable in studying consciousness by analyzing brain activity patterns associated with different states of consciousness.",
    "what are alpha waves in EEG": "Alpha waves are brainwaves with a frequency range of 8 to 12 Hz, commonly observed during relaxed wakefulness with eyes closed.",
    "how does EEG help in understanding seizures": "EEG recordings during seizures provide insights into the specific brain regions involved and the type of seizure activity, aiding in diagnosis and treatment planning.",
    "is EEG useful in diagnosing anxiety disorders": "EEG is not routinely used for diagnosing anxiety disorders. Other psychological and diagnostic assessments are more commonly employed.",
    "what are delta waves in EEG": "Delta waves are brainwaves with a frequency range of 0.5 to 4 Hz, typically observed during deep sleep and certain brain disorders.",
    "how is EEG helpful in studying brain development in children": "EEG is used to study brain maturation and changes in brain activity patterns during different developmental stages in children.",
    "what are theta waves in EEG": "Theta waves are brainwaves with a frequency range of 4 to 8 Hz, commonly seen during light sleep and periods of drowsiness.",
    "is EEG used in diagnosing autism spectrum disorder (ASD)": "EEG is not a primary diagnostic tool for ASD. However, it can contribute to the overall assessment of brain function in individuals with ASD.",
    "what is the difference between EEG and fMRI": "EEG measures the brain's electrical activity, while fMRI (functional magnetic resonance imaging) measures changes in blood flow in the brain, providing information about brain regions involved in specific tasks.",
    "can EEG help in predicting neurodevelopmental disorders in infants": "EEG can be used in certain cases to assess brain development and detect early signs of neurodevelopmental disorders in infants.",
    "what are the challenges in using EEG for psychiatric disorders": "Challenges in EEG include artifact contamination, variability in interpretations, and the need for specialized training in EEG analysis.",
    "how is EEG used in studying brain responses to stimuli": "EEG is used to study event-related potentials (ERPs) in response to various stimuli, providing insights into cognitive processes and brain functioning.",
    "is EEG used in diagnosing multiple sclerosis (MS)": "EEG is not a primary diagnostic tool for MS. Imaging techniques and other tests are more commonly used for MS diagnosis.",
    "what is the role of EEG in studying memory": "EEG is used in memory research to study brainwave patterns associated with different memory processes, such as encoding and retrieval.",
    "can EEG help in diagnosing schizophrenia": "EEG is not used as a standalone diagnostic tool for schizophrenia. However, it can provide supplementary information in the assessment of schizophrenia.",
    "how is EEG used in studying brain activity during meditation": "EEG is used to study changes in brainwave patterns associated with meditation, providing insights into the brain's response to meditative practices.",
    "is EEG used in diagnosing bipolar disorder": "EEG is not routinely used for diagnosing bipolar disorder. Other clinical and psychological assessments are more commonly employed.",
    "what are the challenges in using EEG for diagnosing epilepsy": "Challenges in EEG for epilepsy include capturing the seizure activity during the recording and interpreting the EEG patterns accurately.",
    "how is EEG helpful in studying brain-computer interfaces (BCIs)": "EEG is used in BCIs to detect brain activity patterns associated with specific intentions or actions, allowing individuals to control external devices with their thoughts.",
    "can EEG detect brain damage": "EEG can provide insights into brain function and detect abnormal patterns associated with brain damage, but it is not a direct tool for detecting brain lesions.",
    "what are the ethical considerations in using EEG for psychiatric research": "Ethical considerations in EEG research include obtaining informed consent, ensuring participant confidentiality, and responsible use of the collected data.",
    "how is EEG used in studying cognitive processes": "EEG is used to study cognitive processes such as attention, memory, language, and decision-making, by examining brainwave patterns associated with these functions.",
    "is EEG useful in diagnosing obsessive-compulsive disorder (OCD)": "EEG is not typically used for diagnosing OCD. Other clinical assessments are more commonly employed in OCD diagnosis.",
    "what are the applications of EEG in neuromarketing": "EEG is used in neuromarketing to study consumer responses to advertisements and products, providing insights into consumer preferences and decision-making processes.",
    "how is EEG used in studying brain plasticity": "EEG is used to study changes in brainwave patterns associated with brain plasticity, such as after learning a new skill or following brain injury.",
    "is EEG used in diagnosing post-traumatic stress disorder (PTSD)": "EEG is not routinely used for diagnosing PTSD. Other psychological assessments and diagnostic criteria are more commonly employed.",
    "what are the advantages of using EEG in brain research": "Advantages of EEG include its non-invasive nature, high temporal resolution, and the ability to study brain activity during real-time tasks.",
    "how is EEG used in studying brain aging": "EEG is used to study age-related changes in brainwave patterns and brain activity associated with cognitive decline and aging.",
    "is EEG helpful in diagnosing attention disorders in children": "EEG is not a primary diagnostic tool for attention disorders in children. Clinical assessments and behavioral evaluations are more commonly used.",
    "what are the applications of EEG in gaming and virtual reality": "EEG is used in gaming and virtual reality to detect player engagement and tailor the gaming experience based on the player's brain responses.",
    "how do I cope with anxiety": "Coping with anxiety involves various techniques such as deep breathing, mindfulness, exercise, and seeking support from friends or a therapist.",
    "what are the symptoms of bipolar disorder": "Bipolar disorder includes mood swings from depressive lows to manic highs. Common symptoms include extreme mood changes, energy fluctuations, and changes in sleep patterns.",
    "how can I help someone with schizophrenia": "Supporting someone with schizophrenia involves listening, encouraging treatment, and learning about the condition to provide understanding and empathy.",
    "what treatments are available for OCD": "Obsessive-Compulsive Disorder (OCD) can be treated with therapies like Cognitive Behavioral Therapy (CBT) and medications prescribed by a psychiatrist.",
    "how can I manage stress": "Managing stress involves self-care practices like exercise, meditation, hobbies, and setting boundaries to avoid burnout.",
    "what is social anxiety": "Social anxiety is an intense fear of social situations. Treatment options include therapy and exposure techniques.",
    "what is PTSD": "Post-Traumatic Stress Disorder (PTSD) can result from traumatic events. Treatment often includes therapy like EMDR and medication.",
    "how can I improve my sleep": "Improving sleep involves establishing a consistent sleep routine, creating a relaxing bedtime environment, and avoiding stimulants close to bedtime.",
    "tell me about ADHD": "Attention-Deficit/Hyperactivity Disorder (ADHD) is a neurodevelopmental disorder. Treatment may involve behavioral therapy and medication.",
    "how can I boost my mood": "Boosting your mood can be achieved through activities you enjoy, spending time with loved ones, and practicing gratitude.",
    "what is borderline personality disorder": "Borderline Personality Disorder (BPD) involves difficulties in managing emotions and relationships. Treatment includes therapy like Dialectical Behavior Therapy (DBT).",
    "how can I practice mindfulness": "Mindfulness involves staying present in the moment. You can practice it through meditation, deep breathing, or engaging in activities mindfully.",
    "what is generalized anxiety disorder": "Generalized Anxiety Disorder (GAD) involves excessive worry and anxiety. Therapy and medication can be beneficial in managing GAD.",
    "how do I deal with grief": "Dealing with grief involves allowing yourself to feel emotions, seeking support from others, and considering grief counseling.",
    "what is an eating disorder": "Eating disorders are mental health conditions involving unhealthy eating habits. Treatment often includes therapy and nutritional counseling.",
    "how can I improve my self-esteem": "Improving self-esteem involves practicing self-compassion, challenging negative thoughts, and setting realistic goals.",
    "tell me about autism": "Autism spectrum disorder (ASD) is a developmental disorder. Early intervention and therapy can support individuals with ASD.",
    "how do I manage panic attacks": "Managing panic attacks may involve deep breathing exercises, grounding techniques, and therapy to address underlying anxiety.",
    "what is postpartum depression": "Postpartum depression occurs after childbirth. Seeking help from healthcare professionals is crucial for treatment.",
    "how can I build resilience": "Building resilience involves fostering positive coping skills, maintaining social connections, and seeking support when needed.",
    "tell me about cognitive-behavioral therapy": "Cognitive-Behavioral Therapy (CBT) is a widely used therapy for various mental health conditions. It focuses on changing negative thought patterns and behaviors.",
    "how can I support someone with an eating disorder": "Supporting someone with an eating disorder involves offering non-judgmental support and encouraging professional help.",
    "what is dissociative identity disorder": "Dissociative Identity Disorder (DID) involves the presence of multiple distinct identities or personalities within one person. Treatment often includes therapy.",
    "how can I practice self-care": "Self-care includes activities that promote physical, emotional, and mental well-being. It can involve hobbies, exercise, and relaxation techniques.",
    "tell me about schizophrenia": "Schizophrenia is a complex mental disorder involving hallucinations and delusions. Treatment may include antipsychotic medication and therapy.",
    "how can I manage anger": "Managing anger involves recognizing triggers, practicing relaxation techniques, and seeking support from therapy.",
    "what is a panic disorder": "Panic disorder involves recurrent panic attacks. Therapy and medications can help manage panic disorder.",
    "how do i cope with a traumatic event": "Coping with a traumatic event involves seeking support from friends, family, or therapy to process emotions and experiences.",
    "tell me about personality disorders": "Personality disorders involve patterns of behavior, emotions, and thinking that deviate from societal norms. Treatment varies based on the specific disorder.",
    "how can i handle workplace stress": "Handling workplace stress involves setting boundaries, prioritizing tasks, and seeking support from colleagues or supervisors.",
    "what is social isolation": "Social isolation is the lack of social interactions. Engaging in social activities and reaching out to others can help reduce social isolation.",
    "how can I deal with intrusive thoughts": "Dealing with intrusive thoughts may involve challenging their validity, practicing grounding techniques, and seeking support from therapy.",
    "tell me about self-harm": "Self-harm involves deliberate injury to oneself. If you or someone you know is self-harming, it's essential to seek professional help immediately.",
    "how can I improve my communication skills": "Improving communication skills involves active listening, assertiveness, and empathy.",
    "what is body dysmorphic disorder": "Body Dysmorphic Disorder (BDD) involves a preoccupation with perceived flaws in appearance. Treatment includes therapy and, in some cases, medication.",
    "how can I manage obsessive thoughts": "Managing obsessive thoughts may involve exposure and response prevention therapy, as well as mindfulness practices.",
    "tell me about substance use disorders": "Substance use disorders involve problematic use of drugs or alcohol. Treatment may include therapy and rehabilitation programs.",
    "how can I build healthy relationships": "Building healthy relationships involves communication, respect, and setting boundaries.",
    "what is a mood disorder": "Mood disorders involve significant changes in mood. Treatment may include therapy and medication.",
    "how can I handle performance anxiety": "Handling performance anxiety involves preparation, deep breathing exercises, and reframing negative thoughts.",
    "tell me about anorexia nervosa": "Anorexia Nervosa is an eating disorder characterized by restrictive eating. Early intervention and treatment are essential for recovery.",
    "how can I cope with loss": "Coping with loss involves allowing yourself to grieve, seeking support from loved ones, and considering bereavement counseling.",
    "what is body image dissatisfaction": "Body image dissatisfaction is a negative perception of one's body. Working on self-acceptance and seeking support can be beneficial.",
    "how can I manage test anxiety": "Managing test anxiety involves studying in advance, staying organized, and using relaxation techniques before the exam.",
    "tell me about bipolar disorder": "Bipolar Disorder involves mood swings between manic and depressive episodes. Treatment often includes medication and therapy.",
    "how can I reduce stress at work": "Reducing stress at work involves time management, breaks, and open communication with supervisors.",
    "what is agoraphobia": "Agoraphobia is a fear of being in situations where escape might be difficult. Treatment often includes exposure therapy.",
    "how can I improve my mental well-being": "Improving mental well-being involves regular exercise, connecting with others, and seeking help when needed.",
    "tell me about hallucinations": "Hallucinations are sensory perceptions without external stimuli. If experiencing hallucinations, it's crucial to seek medical evaluation.",
    "how can I support a loved one with depression": "Supporting a loved one with depression involves listening, offering assistance, and encouraging them to seek professional help.",
    "what is body-focused repetitive behavior": "Body-Focused Repetitive Behavior (BFRB) involves repetitive behaviors like hair pulling or skin picking. Treatment includes therapy.",
    "how can I manage social phobia": "Managing social phobia involves gradual exposure to social situations and learning coping techniques.",
    "tell me about specific phobias": "Specific phobias involve intense fear of specific objects or situations. Therapy like Cognitive-Behavioral Therapy (CBT) can help."



}






st.set_page_config(
    page_title="Brainwave Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_theme = """
        [theme]
        primaryColor="#19454c"
        backgroundColor="#316371"
        secondaryBackgroundColor="#131743"
        textColor="#ffffff"
        font="serif"
    """

    # Apply the custom theme
st.write(f"<style>{custom_theme}</style>", unsafe_allow_html=True)
model = pickle.load(open("E:/Brainwave Analysis/project/model.pkl", "rb"))


hide_streamlit_style = """
        <style>
        #MainMenu, .stNotification, .stSystemWarning {
            display: none;
        }
        </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
df = pd.read_csv('https://raw.githubusercontent.com/mubeen161/Datasets/main/EEG.machinelearing_data_BRMH.csv')

# Set page configurations
df = df.rename({'sex': 'gender', 'eeg.date': 'eeg date', 'main.disorder': 'main disorder',
                'specific.disorder': 'specific disorder'}, axis=1)
df['age'] = df['age'].round(decimals=0)
df1=df.loc[:,'gender':'specific disorder']
df1=df1.drop('eeg date',axis=1)

def plot_eeg(levels, positions, axes, fig, ch_names=None, cmap='Spectral_r', cb_pos=(0.9, 0.1),cb_width=0.04, cb_height=0.9, marker=None, marker_style=None, vmin=None, vmax=None, **kwargs):
  if 'mask' not in kwargs:
    mask = np.ones(levels.shape[0], dtype='bool')
  else:
    mask = None
  im, cm = mne.viz.plot_topomap(levels, positions, axes=axes, names=ch_names,cmap=cmap, mask=mask, mask_params=marker_style, show=False, **kwargs)

def reformat_name(name):
    splitted = name.split(sep='.')
    if len(splitted) < 5:
        return name
    if splitted[0] != 'COH':
        result = f'{splitted[2]}.{splitted[4]}'
    else:
        result = f'{splitted[0]}.{splitted[2]}.{splitted[4]}.{splitted[6]}'
    return result


df.rename(reformat_name, axis=1, inplace=True)
# st.set_page_config(page_title='Streamlit Dashboard')


# Mean powers per main disorder
main_mean = df.groupby('main disorder').mean().reset_index()
# Mean powers per specific disorder
spec_mean = df.groupby('specific disorder').mean().reset_index()
# List of bands
msd=['Mood disorder','Addictive disorder','Trauma and stress related disorder','Schizophrenia','Anxiety disorder','Healthy control','Obsessive compulsive disorder']
bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']
ssd=['Acute stress disorder','Adjustment disorder','Alcohol use disorder','Behavioral addiction disorder','Bipolar disorder','Depressive disorder','Healthy Control','Obsessive compulsive disorder','Panic disorder','Posttraumatic stress disorder','Schizophrenia','Social anxiety disorder']
# Convert from wide to long
main_mean = pd.wide_to_long(main_mean, bands, ['main disorder'], 'channel', sep='.', suffix='\w+')
spec_mean = pd.wide_to_long(spec_mean, bands, ['specific disorder'], 'channel', sep='.', suffix='\w+')

# Define channels
chs = {'FP1': [-0.03, 0.08], 'FP2': [0.03, 0.08], 'F7': [-0.073, 0.047], 'F3': [-0.04, 0.041],
       'Fz': [0, 0.038], 'F4': [0.04, 0.041], 'F8': [0.073, 0.047], 'T3': [-0.085, 0], 'C3': [-0.045, 0],
       'Cz': [0, 0], 'C4': [0.045, 0], 'T4': [0.085, 0], 'T5': [-0.073, -0.047], 'P3': [-0.04, -0.041],
       'Pz': [0, -0.038], 'P4': [0.04, -0.041], 'T6': [0.07, -0.047], 'O1': [-0.03, -0.08], 'O2': [0.03, -0.08]}
channels = pd.DataFrame(chs).transpose()
# Create a Streamlit app
# Define home page content

def home_page():
    st.title('Brain-Wave Analytics')
    # st.write('This is the home page.')

    # Random text
    st.header('Muffakham Jah College of Engineering and Technology')
  
    
    # Display images
    image_paths = ['eeg.jpg', 'wave.png', 'process.jpg', 'density.png','pie.png','bar.png','chart.png','future.png']
    # List of captions for the images
    captions = ['EEG Recording', 'Types of Brain Waves', 'Process of EEG to ML', 'Density Plot of main disorder','Main Disorder Ratio','Specific Disorder Ratio','Male-Famale Ratio','Wearable Devices Available']
    # Define the number of columns in the grid
    num_columns = 2
    # Calculate the number of rows based on the number of images and columns
    num_rows = len(image_paths) // num_columns
    # Loop over the rows and columns to display the images in a grid
    for row in range(num_rows):
        col1, col2 = st.columns(num_columns)
        for col in [col1, col2]:
            if image_paths and captions:
                image_path = image_paths.pop(0)
                caption = captions.pop(0)
                image = Image.open(image_path)
                col.image(image, caption=caption, use_column_width=True)





def plot_wave_comparison(df, x_axis, y_axis):
    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot x-axis and y-axis
    ax1.plot(df[x_axis])
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title(x_axis)

    ax2.plot(df[y_axis])
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title(y_axis)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    st.pyplot(fig)


# Define Prediction page content
def prediction_page():
    # st.title('Prediction')
    # Add code for prediction here
    st.title("Disorder Prediction Based on EEG data")
    st.text_input("Enter EEG Data in numericial format :")
    data=pd.read_csv('https://raw.githubusercontent.com/mubeen161/Datasets/main/EEG.machinelearing_data_BRMH.csv')
    data=data.rename(columns={"specific.disorder": "sd", "main.disorder": "md"})
    data = data.fillna(0)
    st.text("OR")
    # Preprocess the data
    md = LabelEncoder()
    data['md'] = md.fit_transform(data['md'])
    sex=LabelEncoder()
    data['sex'] = sex.fit_transform(data['sex'])
    data['sd'] = md.fit_transform(data['sd'])
    data = data.drop(['eeg.date', 'no.'], axis=1)
    data=data.round(4)
    X = data.drop('sd', axis=1)
    y = data['sd']
    X = X.round(3)
    X["age"] = X["age"].round(0)
    selected_data = st.selectbox("Select a person's Data from Dataset :", X.index)

    # Retrieve the selected row from X_test
    input_array = X.loc[selected_data].values

    # Make prediction on user input
    if st.button("Predict"):
        features = np.array(input_array).reshape(1, -1)
        prediction = model.predict(features)
        t = prediction[0]

        if t == 0:
            ls = "Acute Stress Disorder"
        elif t == 1:
            ls = "Adjustment Disorder"
        elif t == 2:
            ls = "Alchohol Use Disorder"
        elif t == 3:
            ls = "Bipolar Disorder"
        elif t == 4:
            ls = "Behavioral Addictive Disorder"
        elif t == 5:
            ls = "Depressive Disorder"
        elif t == 6:
            ls = "Healthy Control"
        elif t == 7:
            ls = "Obsessive Compulsive Disorder"
        elif t == 8:
            ls = "Panic Disorder"
        elif t == 9:
            ls = "Post Traumatic Stress Disorder"
        elif t == 10:
            ls = "Schizophrenia"
        elif t == 11:
            ls = "Social Anxiety Disorder"
        else:
            ls = "Unknown Disorder"

        st.success(f"This individual has the highest probability of having {ls}")
        st.text("Please note that this result is not entirely reliable, and further medical assessment maybe required for a conclusive diagnosis.")

# Define Plots page content
def plots_page():
    st.title('Plots')
    num_variables = st.sidebar.selectbox("Select number of variables:", [1, 2, 3])

    # Add code for plots here
    if num_variables == 1:
        plot_type1 = st.sidebar.selectbox("Select plot type:", ["Swarmplot", "Histogram", "Bar Plot","Line Plot","Scatter Plot"])
        
        # Select a single variable
        variable = st.sidebar.selectbox("Select a variable:", df1.columns)
        fig, ax = plt.subplots()
        if plot_type1 == "Swarmplot":
            sns.swarmplot(df1[variable])
            ax.set_xlabel(variable)
            # ax.set_ylabel("Index")
            ax.set_title(f"Swarm plot of {variable}")
        
        elif plot_type1 == "Histogram":
            ax.hist(df1[variable])
            ax.set_xlabel(variable)
            # ax.set_ylabel("Index")
            ax.set_title(f"Histogram of {variable}")
        
        elif plot_type1 == "Bar Plot":
            ax.bar(df1.index, df1[variable])
            ax.set_xlabel("Index")
            ax.set_ylabel(variable)
            ax.set_title(f"Bar plot of {variable}")
        elif plot_type1=="Line Plot":
            ax.plot(df1.index, df1[variable])
            plt.xlabel("Index")
            plt.ylabel(variable)
            plt.title("Line Plot of " + variable)
            plt.show()
        elif plot_type1=="Scatter Plot":
            fig, ax = plt.subplots()
            ax.scatter(df1.index, df1[variable])
            ax.set_xlabel("Index")
            ax.set_ylabel(variable)
            ax.set_title(f"Scatter Plot of {variable}")
        elif plot_type1 == "Violin Plot":
            ax.violinplot(df1[variable])
            ax.set_xlabel(variable)
            ax.set_title(f"Violin plot of {variable}")
    
        
        st.pyplot(fig)
    

    elif num_variables == 2:
        # Select two variables
        plot_type = st.sidebar.selectbox("Select plot type:", ["Bar Plot","Line Plot","Scatter Plot","Violin Plot","Kde Plot","Hexbin Plot","Area Plot"])
        x_variable = st.sidebar.selectbox("Select the x-axis variable:", df1.columns)
        y_variable = st.sidebar.selectbox("Select the y-axis variable:", df1.columns)
        fig, ax = plt.subplots()
    
        if plot_type == "Scatter Plot":
            ax.scatter(df1[x_variable], df1[y_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Scatter plot of {x_variable} vs {y_variable}")
        
        elif plot_type == "Line Plot":
            ax.plot(df1[x_variable], df1[y_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Line plot of {x_variable} vs {y_variable}")
        
        elif plot_type == "Bar Plot":
            ax.bar(df1[x_variable], df1[y_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Bar plot of {x_variable} vs {y_variable}")
    
    
        elif plot_type == "Violin Plot":
            sns.violinplot(x=df1[x_variable], y=df1[y_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Violin plot of {x_variable} and {y_variable}")


    
        elif plot_type == "Kde Plot":
            sns.kdeplot(data=df1, x=df1[x_variable], y=df1[y_variable], shade=True)
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Kde plot of {x_variable} and {y_variable}")
        elif plot_type == "Hexbin Plot":
            ax.hexbin(df1[x_variable], df1[y_variable], gridsize=20, cmap='viridis')
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Hexbin plot of {x_variable} and {y_variable}")
        elif plot_type == "Area Plot":
            ax.fill_between(df1[x_variable], df1[y_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_title(f"Area plot of {x_variable} and {y_variable}")
    
        st.pyplot(fig)
    elif num_variables == 3:
        # Select three variables
        plot_type = st.sidebar.selectbox("Select plot type:", ["Scatter Plot", "Bar Plot", "Line Plot"])
        x_variable = st.sidebar.selectbox("Select the x-axis variable:", df.columns)
        y_variable = st.sidebar.selectbox("Select the y-axis variable:", df.columns)
        z_variable = st.sidebar.selectbox("Select the z-axis variable:", df.columns)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    
        if plot_type == "Scatter Plot":
            ax.scatter3D(df[x_variable], df[y_variable], df[z_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_zlabel(z_variable)
            ax.set_title(f"Scatter plot of {x_variable}, {y_variable}, {z_variable}")
    
    
        elif plot_type == "Bar Plot":
            ax.bar3d(df[x_variable], df[y_variable], 0, 1, 1, df[z_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_zlabel(z_variable)
            ax.set_title(f"Bar plot of {x_variable}, {y_variable}, {z_variable}")
    
        elif plot_type == "Line Plot":
            ax.plot3D(df[x_variable], df[y_variable], df[z_variable])
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)
            ax.set_zlabel(z_variable)
            ax.set_title(f"Line plot of {x_variable}, {y_variable}, {z_variable}")
    
    
        
        st.pyplot(fig)

# Define Wave Compare page content
def wave_compare_page():
    st.title('Wave Comparison')
    image_path = "head.png"
    
    # Open the image file using PIL
    image = Image.open(image_path)
    resized_image = image.resize((8, 8))
    
    # Calculate the center alignment
    image_width, image_height = resized_image.size
    
    # Display the resized image with center alignment
    st.sidebar.image(image, width=image_width, caption='Electrode Position Image', use_column_width=True, output_format='PNG')

    df1=df.drop(['no.','gender','age','eeg date','education','IQ','main disorder','specific disorder'],axis=1)
    x_axis = st.selectbox('Select Wave 1 : ', df1.columns)
    y_axis = st.selectbox('Select Wave 2 : ', df1.columns)

    # Check if the DataFrame is not empty
    if not df.empty:
        # Call the function to plot the wave comparison
        plot_wave_comparison(df, x_axis, y_axis)
    else:
        st.write('No data available to plot.')

def topographic_brain_activity():
    # st.write("Topographic Brain Activity selected")
    # Add your code for Topographic Brain Activity functionality here
    st.title("EEG Data Analysis")
    st.title('Brain Compare')
    img_pt='level.png'
    # Add code for brain compare here
    image = Image.open(img_pt)
    image_width, image_height = image.size
    
    # Display the resized image with center alignment
    st.sidebar.image(image, width=image_width, caption='Brainwave Intensity Representation', use_column_width=True, output_format='PNG')
    test = spec_mean.loc[st.selectbox("Disorder",ssd), st.selectbox("Type of Brain wave",bands)]
    # Display the EEG plot
    fig, ax = plt.subplots()
    plot_eeg(test, channels.to_numpy(), ax, fig, marker_style={'markersize': 4, 'markerfacecolor': 'black'})
    st.pyplot(fig)

def disorder_comparison():
    st.write("Disorder Comparison selected")
    # Add your code for Disorder Comparison functionality here
    test_schizo = main_mean.loc[st.selectbox("Disorder 1",msd), st.selectbox("Bnads 1",bands)]
    test_control = main_mean.loc[st.selectbox("Disorder 2",msd), st.selectbox("Bnads 2",bands)]
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
   
    # Plot the first subplot
    plot_eeg(test_schizo, channels.to_numpy(), ax1, fig, marker_style={'markersize': 4, 'markerfacecolor': 'black'})
    ax1.set_title('Disorder 1')
   
    # Plot the second subplot
    plot_eeg(test_control, channels.to_numpy(), ax2, fig, marker_style={'markersize': 4, 'markerfacecolor': 'black'})
    ax2.set_title('Disorder 2')
   
    # Display the plot
    st.pyplot(fig)

def brain_simulation():
    st.title("Brain Simulation selected")
    video_url = "plot.mp4"
    st.video(video_url)
# Define Brain Compare page content
def brain_compare_page():
    st.title('Brain Compare')
    img_pt='level.png'
    # Add code for brain compare here
    image = Image.open(img_pt)
    image_width, image_height = image.size
    
    # Display the resized image with center alignment
    st.sidebar.image(image, width=image_width, caption='Brainwave Intensity Representation', use_column_width=True, output_format='PNG')
    if st.button("Topographic Brain Activity"):
        topographic_brain_activity()

    if st.button("Disorder Comparison"):
        disorder_comparison()

    if st.button("Brain Simulation"):
        brain_simulation()


# Define Stress Level page content
def stress_level_page():
    st.title('Stress Level')
    # Add code for stress level here
    column_name = st.selectbox("Select Band",bands)
    highest_value = main_mean[column_name].max()
    lowest_value = main_mean[column_name].min()
    

    min_value = round(lowest_value, 2)
    max_value = round(highest_value, 2)


    fig, ax = plt.subplots(figsize=(6, 1))
    ax.set_axis_off()


    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(min_value, max_value)


    ax.imshow(np.arange(min_value, max_value).reshape(1, -1), cmap=cmap, norm=norm, aspect='auto')
    ax.text(0, 0, str(min_value), ha='left', va='center', color='black', weight='light')
    ax.text(max_value - min_value, 0, str(max_value), ha='right', va='center', color='black', weight='light')
    plt.title('Stress level using beta brain waves')
    st.pyplot(fig)
    # st.write("Brain Simulation selected")
    img_path = "stresslevel.jpg"
    img=Image.open(img_path)
    # Display the GIF image
    st.image(img,caption='Reference Chart' ,use_column_width=True)

    # Display the plot using Streamlit
    

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=120, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def chat():
    st.title("AI ASSISTANT")
    st.write("Welcome to AI Guide...!!!")

    while True:
        try:
            user_input = st.text_input("You: ", "").lower()  # Convert user input to lowercase

            # Check if the user input has a predefined response
            if user_input in healthcare_responses:
                st.write("AI bot:", healthcare_responses[user_input])
            elif user_input.lower() == "exit":
                break
            else:
                # If no predefined response is found, use the GPT-2 model
                prompt = f"You: {user_input}\nAI bot: "
                response = generate_response(prompt)
                st.write("AI bot:", response)

        except Exception as e:
            # Handle the exception here, you can log it or display a custom error message.
            st.write("")


# Main code
def main():
    # Dropdown menu for page selection
    page_options = {
        'Home': home_page,
        'Plots': plots_page,
        'Wave Compare': wave_compare_page,
        # 'Brain Compare': brain_compare_page,
        'Stress Level': stress_level_page,
        'Topographic Brain Activity':topographic_brain_activity,
        'Disorder Comparison':disorder_comparison,
        'Brain Simulation':brain_simulation,
        'Prediction': prediction_page,
        'AI - Assistant ':chat
    }
    selected_page = st.sidebar.selectbox('Select a page', list(page_options.keys()))
    page = page_options[selected_page]

    # Display selected page content
    page()


if __name__ == '__main__':
    main()
