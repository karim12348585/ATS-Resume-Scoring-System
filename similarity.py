from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer 

from sklearn.metrics.pairwise import cosine_similarity
import re


from data_extraction import get_gemini_response


def asymmetric_jaccard(required_skills, candidate_skills):
    """
    Calculate asymmetric Jaccard similarity
    Focuses on how many required skills are covered by candidate skills
    """
    if not required_skills:
        return 1.0  # If no required skills specified, candidate meets all requirements

    # Convert to lowercase for case-insensitive matching
    required_lower = [skill.lower() for skill in required_skills]
    candidate_lower = [skill.lower() for skill in candidate_skills]

    # Count matches (skills that are both required and possessed by candidate)
    matches = sum(1 for skill in required_lower if any(re.search(r'\b' + re.escape(skill) + r'\b', cand_skill) or
                                                       re.search(r'\b' + re.escape(cand_skill) + r'\b', skill)
                                                       for cand_skill in candidate_lower))

    # Asymmetric Jaccard: matches / total required skills
    return matches / len(required_lower) if len(required_lower) > 0 else 0.0


def calculate_cosine_similarity(text1, text2, model):
    """Calculate cosine similarity between two texts using SBERT embeddings"""
    # Embed the texts
    embedding1 = model.encode([text1])[0]
    embedding2 = model.encode([text2])[0]

    # Calculate cosine similarity
    similarity = cosine_similarity([embedding1] , [embedding2])[0][0]
    return similarity


def experience_penalty(required_years, candidate_years):
    """
    Calculate penalty for missing years of experience
    Returns a score between 0 and 1, where 1 means meets or exceeds requirements
    """
    if required_years <= 0:
        return 1.0  # No experience required

    if candidate_years >= required_years:
        return 1.0  # Meets or exceeds requirements

    # Linear penalty for missing years
    ratio = candidate_years / required_years
    return max(0, min(1 , ratio))  # Ensure score is between 0 and 1


def calculate_final_score(skills_score, summary_score, experience_score):
    """
    Calculate final score with weighted components
    Weights: skills 40%, summary 30%, experience 30%
    """
    weights = {
        'skills': 0.4,
        'summary': 0.3,
        'experience': 0.3
    }

    final_score = (
            skills_score * weights['skills'] +
            summary_score * weights['summary'] +
            experience_score * weights['experience']
    )

    return final_score


def generate_detailed_analysis(resume_data, job_data, scores):
    """Generate detailed analysis and recommendations using LLM"""

    # Create summary of scores for the LLM
    scores_summary = f"""
    Skills Match Score: {scores['skills_score']:.2f} (Matching {scores['matching_skills']} out of {scores['total_required_skills']} required skills)
    Missing Skills: {', '.join(scores['missing_skills'])}
    Experience Score: {scores['experience_score']:.2f} (Candidate: {resume_data['years_experience']} years, Required: {job_data['required_years']} years)
    Resume-Job Summary Relevance: {scores['summary_score']:.2f}
    Overall Match: {scores['final_score']:.2f} (or {int(scores['final_score'] * 100)}%)
    """

    # Create prompt for LLM analysis
    analysis_prompt = f"""
    You are an expert ATS and HR consultant. Based on the following data, provide a detailed analysis of this candidate's fit for the job:

    JOB SUMMARY:
    {job_data['summary']}

    CANDIDATE SUMMARY:
    {resume_data['summary']}

    SCORING BREAKDOWN:
    {scores_summary}

    Provide a professional analysis covering:
    1. Overall assessment of candidate's fit (2-3 sentences)
    2. Strengths - what makes this candidate qualified (2-3 bullet points)
    3. Gaps - what skills or experiences the candidate is missing (2-3 bullet points)
    4. Recommendations - specific advice to improve resume for this role (2-3 bullet points)

    Keep the analysis factual, helpful and constructive.
    """

    analysis = get_gemini_response(analysis_prompt)
    return analysis