from dotenv import load_dotenv

import streamlit as st
import os

import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

from data_extraction import extract_resume_data, extract_job_data, input_pdf_setup, extract_technologies
from similarity import asymmetric_jaccard, calculate_cosine_similarity, experience_penalty, calculate_final_score, \
    generate_detailed_analysis

# Load environment variables
load_dotenv()
api_key = os.getenv("api_key")

genai.configure(api_key=api_key)

# Model for embedding text
model = SentenceTransformer('all-MiniLM-L6-v2')


# Streamlit UI
st.set_page_config(page_title="Advanced ATS Resume Analysis", layout="wide")
st.title("Advanced ATS Resume Analysis System")

# Set up two columns for input
col1, col2 = st.columns(2)

with col1:
    st.header("Job Description")
    job_description = st.text_area("Paste the job description here", height=300)

with col2:
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])
    
    if uploaded_file is not None:
        st.success("Resume uploaded successfully!")

# Process button
if st.button("Analyze Resume Match"):
    if uploaded_file is not None and job_description:
        with st.spinner("Analyzing resume and job description..."):
            try:
                # Process PDF
                pdf_content = input_pdf_setup(uploaded_file)
                
                # Extract data
                resume_data = extract_resume_data(pdf_content, job_description)
                job_data = extract_job_data(job_description)
                
                # Convert text skills to lists
                resume_skills = extract_technologies(resume_data['skills'])
                job_skills = extract_technologies(job_data['required_skills'])
                
                # Calculate scores
                skills_score = asymmetric_jaccard(job_skills, resume_skills)
                summary_score = calculate_cosine_similarity(resume_data['summary'], job_data['summary'],model)
                experience_score = experience_penalty(job_data['required_years'], resume_data['years_experience'])
                final_score = calculate_final_score(skills_score, summary_score, experience_score)
                
                # Calculate missing skills
                job_skills_lower = [skill.lower() for skill in job_skills]
                resume_skills_lower = [skill.lower() for skill in resume_skills]
                missing_skills = [skill for skill in job_skills if skill.lower() not in 
                                [s for s in resume_skills_lower for j in job_skills_lower 
                                  if re.search(r'\b' + re.escape(j) + r'\b', s)]]
                
                # Prepare scores dictionary for detailed analysis
                scores = {
                    'skills_score': skills_score,
                    'summary_score': summary_score,
                    'experience_score': experience_score,
                    'final_score': final_score,
                    'matching_skills': len(job_skills) - len(missing_skills),
                    'total_required_skills': len(job_skills),
                    'missing_skills': missing_skills
                }
                
                # Generate detailed analysis
                detailed_analysis = generate_detailed_analysis(resume_data, job_data, scores)
                
                # Display results
                st.header("Analysis Results")
                
                # Score display
                st.subheader("Match Score")
                score_percentage = int(final_score * 100)
                
                # Create score chart
                score_color = "green" if score_percentage >= 70 else "orange" if score_percentage >= 50 else "red"
                
                # Score gauge chart using Altair
                score_data = pd.DataFrame({
                    'category': ['Skills Match', 'Experience Match', 'Content Relevance', 'Overall Score'],
                    'value': [skills_score, experience_score, summary_score, final_score]
                })
                
                # Create a color scale for the scores
                def get_color(score):
                    if score >= 0.7:
                        return "green"
                    elif score >= 0.5:
                        return "orange"
                    else:
                        return "red"
                
                # Add color column to the dataframe
                score_data['color'] = score_data['value'].apply(get_color)
                
                score_chart = alt.Chart(score_data).mark_bar().encode(
                    x=alt.X('value:Q', scale=alt.Scale(domain=[0, 1]), title='Score (0-1)'),
                    y=alt.Y('category:N', title=None),
                    color=alt.Color('color:N', scale=None),
                    tooltip=['category', alt.Tooltip('value:Q', format='.2f')]
                ).properties(
                    title='Resume Match Scores',
                    width=600,
                    height=200
                )
                
                st.altair_chart(score_chart, use_container_width=True)
                
                # Skills comparison
                st.subheader("Skills Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Required Skills:**")
                    for skill in job_skills:
                        if skill in missing_skills:
                            st.markdown(f"❌ {skill}")
                        else:
                            st.markdown(f"✅ {skill}")
                
                with col2:
                    st.write("**Resume Skills:**")
                    for skill in resume_skills:
                        st.markdown(f"• {skill}")
                
                # Experience comparison
                st.subheader("Experience Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Required Experience:** {job_data['required_years']} years")
                
                with col2:
                    st.write(f"**Candidate Experience:** {resume_data['years_experience']} years")
                    if resume_data['years_experience'] < job_data['required_years']:
                        st.write(f"*Missing {job_data['required_years'] - resume_data['years_experience']:.1f} years of required experience*")
                
                # Display detailed analysis
                st.subheader("Expert Analysis")
                st.write(detailed_analysis)
                
                # Show explanation of scoring method
                with st.expander("How is the score calculated?"):
                    st.markdown("""
                    **Scoring Methodology:**
                    
                    1. **Skills Match (40% of final score):**
                       - Uses asymmetric Jaccard similarity to compare required skills to candidate skills
                       - Focuses on what percentage of required skills the candidate possesses
                       
                    2. **Summary Relevance (30% of final score):**
                       - Uses SBERT (Sentence-BERT) embeddings to convert resume and job summaries into vectors
                       - Calculates cosine similarity between these vectors to measure content relevance
                       
                    3. **Experience Match (30% of final score):**
                       - Compares required years of experience with candidate's experience
                       - Full score if candidate meets or exceeds requirements, proportional penalty otherwise
                       
                    **Final Score:** Weighted average of the three components above
                    """)
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
    else:
        st.warning("Please upload a resume and provide a job description.")