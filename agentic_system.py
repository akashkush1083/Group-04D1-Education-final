"""
Simple Agentic Education System with Research and Writing Agents
Uses langchain-groq without CrewAI dependency (works better with Python 3.14)
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=2000
)

def research_agent(topic):
    """
    Research Agent - Collects and structures information
    """
    research_prompt = f"""
    You are an expert education researcher. Research the topic: {topic}
    
    Your research should include:
    1. Clear definition and explanation
    2. Key concepts and terminology
    3. Important details and nuances
    4. Real-world applications
    5. Common misconceptions
    6. Historical context (if relevant)
    
    Structure your findings in a clear, organized manner with proper sections.
    Be comprehensive and detailed.
    """
    
    response = llm.invoke(research_prompt)
    return response.content

def writing_agent(research_content, topic):
    """
    Writing Agent - Creates well-formatted educational content
    """
    writing_prompt = f"""
    You are a skilled educational writer. Using the following research, create a comprehensive study guide for: {topic}
    
    Research findings:
    {research_content}
    
    The study guide should include:
    1. Clear introduction with definition
    2. Key concepts section with bullet points
    3. Detailed explanation with examples
    4. Real-world applications
    5. Common mistakes to avoid
    6. Quick summary/cheat sheet
    7. Practice questions (optional)
    
    Use Markdown formatting with:
    - Headers (##, ###)
    - Bullet points
    - Bold text for emphasis
    - Code blocks where appropriate
    - Tables for comparisons
    
    Make it engaging, clear, and comprehensive.
    """
    
    response = llm.invoke(writing_prompt)
    return response.content

def create_study_guide(topic):
    """
    Create a comprehensive study guide using agentic workflow
    """
    try:
        # Step 1: Research
        research_content = research_agent(topic)
        
        # Step 2: Write
        final_content = writing_agent(research_content, topic)
        
        return final_content
    except Exception as e:
        # If agentic fails, return simple direct response
        print(f"Agentic error: {e}")
        simple_prompt = f"""
        Create a comprehensive study guide for: {topic}
        
        Include:
        1. Clear definition
        2. Key concepts (bullet points)
        3. Detailed explanation
        4. Real-world examples
        5. Common mistakes
        6. Quick summary
        
        Use Markdown formatting.
        """
        response = llm.invoke(simple_prompt)
        return response.content

def create_explanation(prompt):
    """
    Create detailed explanation using agentic workflow
    """
    try:
        # Research phase
        research_prompt = f"""
        You are an expert researcher. Research the topic based on this prompt: {prompt}
        
        Focus on:
        1. Understanding the core question/request
        2. Gathering relevant information
        3. Identifying key points to address
        4. Finding supporting examples
        
        Provide comprehensive research findings.
        """
        
        research_response = llm.invoke(research_prompt)
        research_content = research_response.content
        
        # Writing phase
        writing_prompt = f"""
        You are a skilled educational writer. Create a comprehensive explanation based on the research for: {prompt}
        
        Research findings:
        {research_content}
        
        Structure:
        1. Direct answer to the question
        2. Detailed explanation
        3. Supporting examples
        4. Key takeaways
        5. Additional context if needed
        
        Use clear, engaging language with Markdown formatting.
        """
        
        writing_response = llm.invoke(writing_prompt)
        return writing_response.content
    except Exception as e:
        # If agentic fails, return simple direct response
        print(f"Agentic error: {e}")
        simple_prompt = f"""
        Create a comprehensive educational explanation for: {prompt}
        
        Provide:
        1. Clear introduction
        2. Detailed explanation with examples
        3. Key takeaways
        4. Practical applications
        
        Format in Markdown with clear sections.
        """
        response = llm.invoke(simple_prompt)
        return response.content
