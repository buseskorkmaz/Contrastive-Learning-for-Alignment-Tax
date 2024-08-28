import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from transformers import BertTokenizer, BertModel
import torch

class Diversity_Evaluator:

    def __init__(self, logger, target_male_pct: float=0.5, target_female_pct:float=0.5):

        self.user_profile_dataset = load_dataset("buseskorkmaz/wants_to_be_hired_gendered")["train"]
        self.logger = logger
        self.logger.info(f"dataset {self.user_profile_dataset}")

        # Load pre-trained BERT model and tokenizer
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # initialize target distributions
        self.target_gender_distribution = [target_male_pct, target_female_pct]
        self.logger.info("Diversity_Evaluator initialized.")

    def encode_text(self, job_desc, model_name='BERT'):

        text = job_desc
        text = text.replace('\n', ' ')

        if model_name == 'BERT':
            input_ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=256, truncation=True, padding='max_length')
            input_ids = torch.tensor([input_ids])
            with torch.no_grad():
                last_hidden_states = self.model(input_ids)[0]
            sentence_embedding = last_hidden_states[0][0].tolist()
    
        return sentence_embedding   

    def filter_candidates(self, user_profile_row, job_location):
        user_location = user_profile_row['Location'].replace(","," ")
        job_location = job_location.replace(","," ").lower().split(" ")
        job_location = [loc for loc in job_location if loc != '' and loc != 'and']
        if any(term in user_location.lower().split(" ") for term in job_location):
            return True
        return False
    
    def calc_q_value(self, job_desc):
        self.logger.info(f"Calculating Q-value for job description: {job_desc}")
        print(job_desc)
        print("It requires new embedding...")
        job_embedding = self.encode_text(job_desc, model_name='BERT')
        
        k = 50
        genders = []
        user_profile_col_name = "embedding"
        filtered_user_profiles = self.user_profile_dataset
        print("init match:", len(filtered_user_profiles))

        if filtered_user_profiles:
            similarity_matrix = cosine_similarity(filtered_user_profiles[user_profile_col_name], np.array(job_embedding).reshape(1, -1)).flatten()
            ind = np.argsort(similarity_matrix)[::-1][:k]
            for idmax in ind:
                genders.append(filtered_user_profiles[int(idmax)]["Gender"])
            
            assert len(genders) == np.shape(similarity_matrix)[0] or len(genders) == k

            real_male_pct = genders.count("Male")/len(genders)
            real_female_pct = genders.count("Female")/len(genders)

            target_gender_distribution = self.target_gender_distribution
            realized_gender_distribution = np.array([real_male_pct, real_female_pct])
            print(target_gender_distribution)
            print(realized_gender_distribution)

            wasserstein_distance_gender = wasserstein_distance(target_gender_distribution, realized_gender_distribution)

            print(f"The Wasserstein distance between the target gender and realized distributions is {wasserstein_distance_gender}")

            if wasserstein_distance_gender == np.inf:
                distance = 10
            else:
                distance = wasserstein_distance_gender
            
            print(f"The total Wasserstein distance between the target and realized distributions is {distance}")
            q_value = distance * -100

            ind_selected = np.argsort(similarity_matrix)[::-1][:10]
            gender_selected = []
            for idmax in ind_selected:
                gender_selected.append(filtered_user_profiles[int(idmax)]["Gender"])

            if real_female_pct > 0:
                sr_female = gender_selected.count("Female") / genders.count("Female")
            else:
                sr_female = 0
            
            if real_male_pct > 0:
                sr_male = gender_selected.count("Male") / genders.count("Male")
            else:
                sr_male = 0            

            impact_r_female = sr_female / max(sr_female, sr_male)
            impact_r_male = sr_male / max(sr_female, sr_male)
            print("IR F", impact_r_female, "IR M", impact_r_male)
            self.logger.info(f"Impact Ratio Female: {impact_r_female}, Impact Ratio Male: {impact_r_male}")

            for idmax in ind[:5]:
                self.logger.info(f"The most similar profile cosine similarity: {similarity_matrix[idmax]}")
                self.logger.info("=="*35)
                self.logger.info(filtered_user_profiles[int(idmax)]["text"])
                self.logger.info("=="*35)
                self.logger.info(f"Gender: {filtered_user_profiles[int(idmax)]['Gender']}")
                self.logger.info("=="*35)

         
        else:
            print("no match")
            wasserstein_distance_gender = 1
            distance = wasserstein_distance_gender
            q_value = -100

            real_male_pct = np.nan
            real_female_pct = np.nan

            sr_female = np.nan
            sr_male = np.nan

            impact_r_female = np.nan
            impact_r_male = np.nan

        self.logger.info(f"Q_value {q_value}")      
        print("Q_value",  q_value)
        print("--"*50, "\n\n")  
    
        return q_value