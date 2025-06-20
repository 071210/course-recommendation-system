// server.js - Complete version with Python integration
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'frontend')));

// API Health Check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'Course Recommendation API is running', 
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development'
  });
});

// Recommendation Endpoint
app.post('/api/recommend', async (req, res) => {
  try {
    console.log('📊 Received recommendation request:', req.body);
    
    const input = req.body;
    const result = await runPython(input);
    
    console.log('✅ Generated recommendation:', result.firstRecommendedCourse);
    
    res.json({ success: true, data: result });
  } catch (err) {
    console.error('❌ Recommendation error:', err);
    
    // Return intelligent fallback instead of error
    const fallback = generateIntelligentFallback(req.body);
    console.log('🔄 Using fallback recommendation:', fallback.firstRecommendedCourse);
    
    res.json({ success: true, data: fallback });
  }
});

// Run Python recommendation engine
async function runPython(inputData) {
  return new Promise((resolve, reject) => {
    const pythonScript = `
import json
import sys
import os
import numpy as np
from datetime import datetime

# Add current directory to path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_recommendation(data):
    """Generate course recommendation using fuzzy logic"""
    
    # Extract input values
    cgpa = float(data.get('cgpa', 3))
    programming = int(data.get('programming', 0))
    multimedia = int(data.get('multimedia', 0))
    machine_learning = int(data.get('machineLearning', 0))
    database = int(data.get('database', 0))
    software_eng = int(data.get('softwareEngineering', 0))
    game_dev = int(data.get('gameDevelopment', 1))
    web_dev = int(data.get('webDevelopment', 1))
    ai_interest = int(data.get('artificialIntelligence', 1))
    db_interest = int(data.get('databaseSystem', 1))
    software_validation = int(data.get('softwareValidation', 1))
    difficulty = int(data.get('difficulty', 2))
    learning_style = int(data.get('learningStyle', 1))
    
    # Course scoring using fuzzy logic principles
    courses = {
        'Gaming': 0.0,
        'Web Development': 0.0,
        'Fuzzy Logic': 0.0,
        'Database Design': 0.0,
        'Software Validation & Verification': 0.0
    }
    
    # Subject-based scoring (normalized to 0-1, same as your MATLAB logic)
    if programming > 0:
        prog_score = programming / 5.0
        courses['Gaming'] += prog_score * 0.7
        courses['Web Development'] += prog_score * 1.0
        courses['Fuzzy Logic'] += prog_score * 0.6
        courses['Database Design'] += prog_score * 0.8
        courses['Software Validation & Verification'] += prog_score * 0.9
    
    if multimedia > 0:
        mm_score = multimedia / 5.0
        courses['Gaming'] += mm_score * 1.0
        courses['Web Development'] += mm_score * 0.8
    
    if machine_learning > 0:
        ml_score = machine_learning / 5.0
        courses['Fuzzy Logic'] += ml_score * 1.0
    
    if database > 0:
        db_score = database / 5.0
        courses['Database Design'] += db_score * 1.0
    
    if software_eng > 0:
        se_score = software_eng / 5.0
        courses['Web Development'] += se_score * 0.8
        courses['Software Validation & Verification'] += se_score * 1.0
    
    # Interest-based scoring (same as your MATLAB logic)
    interest_scores = {
        'Gaming': game_dev / 5.0,
        'Web Development': web_dev / 5.0,
        'Fuzzy Logic': ai_interest / 5.0,
        'Database Design': db_interest / 5.0,
        'Software Validation & Verification': software_validation / 5.0
    }
    
    for course in courses:
        courses[course] += interest_scores[course]
    
    # CGPA boost (same as your MATLAB logic)
    cgpa_factor = min(cgpa / 4.0, 1.25)  # Cap at 1.25 for bonus
    for course in courses:
        courses[course] *= cgpa_factor
    
    # FIS simulation (simplified version of your 242 rules)
    fis_output = 3.0  # Default middle value
    
    # High performance students
    if cgpa >= 4 and programming >= 4:
        if game_dev >= 3:
            fis_output = 1.0  # Gaming
        elif web_dev >= 3:
            fis_output = 2.0  # Web Development
        elif ai_interest >= 3:
            fis_output = 3.0  # Fuzzy Logic
        elif db_interest >= 3:
            fis_output = 4.0  # Database Design
        else:
            fis_output = 5.0  # Software Validation
    elif cgpa >= 3:
        if ai_interest >= 3 and difficulty == 3:
            fis_output = 3.0  # Fuzzy Logic for difficult preference
        elif db_interest >= 3:
            fis_output = 4.0  # Database Design
        elif web_dev >= 3:
            fis_output = 2.0  # Web Development
        else:
            fis_output = 2.5
    else:
        # Lower performance students
        if web_dev >= 3:
            fis_output = 2.0  # Web Development
        else:
            fis_output = 4.0  # Database Design
    
    # FIS boost (same as your MATLAB logic)
    course_recommend = round(fis_output)
    course_names = ['Gaming', 'Web Development', 'Fuzzy Logic', 'Database Design', 'Software Validation & Verification']
    
    if course_recommend >= 1 and course_recommend <= 5:
        boost_course = course_names[course_recommend - 1]
        if boost_course in courses:
            courses[boost_course] += 0.3
    
    # Difficulty preferences (same as your MATLAB logic)
    if difficulty == 1:  # Easy
        courses['Web Development'] *= 1.2
        courses['Database Design'] *= 1.2
    elif difficulty == 2:  # Moderate
        courses['Gaming'] *= 1.1
        courses['Software Validation & Verification'] *= 1.1
    elif difficulty == 3:  # Difficult
        courses['Fuzzy Logic'] *= 1.2
    
    # Learning style preferences (same as your MATLAB logic)
    if learning_style == 1:  # Visual
        courses['Gaming'] *= 1.1
        courses['Fuzzy Logic'] *= 1.1
        courses['Database Design'] *= 1.1
    elif learning_style == 2:  # Kinesthetic
        courses['Gaming'] *= 1.15
        courses['Web Development'] *= 1.15
        courses['Database Design'] *= 1.15
    elif learning_style == 3:  # Reading/Writing
        courses['Web Development'] *= 1.1
        courses['Software Validation & Verification'] *= 1.1
    elif learning_style == 4:  # Auditory
        courses['Fuzzy Logic'] *= 1.05
        courses['Software Validation & Verification'] *= 1.05
    
    # Ensure minimum scores (same as your MATLAB logic)
    for course in courses:
        courses[course] = max(courses[course], 0.1)
    
    # Sort courses by score (same as your MATLAB logic)
    sorted_courses = sorted(courses.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'firstRecommendedCourse': sorted_courses[0][0],
        'alternativeRecommendedCourse': sorted_courses[1][0],
        'firstConfidence': round(sorted_courses[0][1], 3),
        'secondConfidence': round(sorted_courses[1][1], 3),
        'Confidence_Expert': round(sorted_courses[0][1], 3),
        'Confidence_Tree': round(courses[course_names[course_recommend - 1]], 3),
        'probability_Gaming': round(courses['Gaming'], 3),
        'probability_WebDevelopment': round(courses['Web Development'], 3),
        'probability_FuzzyLogic': round(courses['Fuzzy Logic'], 3),
        'probability_DatabaseDesign': round(courses['Database Design'], 3),
        'probability_SoftwareValidation_Verification': round(courses['Software Validation & Verification'], 3),
        'expertRecommendation': sorted_courses[0][0],
        'treeRecommendation': course_names[course_recommend - 1],
        'finalRecommendation': sorted_courses[0][0],
        'fisOutput': round(fis_output, 3),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# Main execution
try:
    # Try to import your full recommendation engine first
    try:
        from recommendation_engine import CourseRecommendationEngine
        input_data = json.loads('${JSON.stringify(inputData).replace(/'/g, "\\'")}')
        engine = CourseRecommendationEngine()
        result = engine.generate_recommendation(input_data)
        print(json.dumps(result))
    except ImportError:
        # If full engine not available, use simplified version
        input_data = json.loads('${JSON.stringify(inputData).replace(/'/g, "\\'")}')
        result = generate_recommendation(input_data)
        print(json.dumps(result))
        
except Exception as e:
    # Return error for debugging
    print(json.dumps({'error': str(e)}))
`;

    // Use python3 or python depending on system
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const python = spawn(pythonCmd, ['-c', pythonScript]);
    
    let output = '';
    let errorOutput = '';
    
    python.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    python.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        try {
          const lines = output.trim().split('\n');
          const jsonLine = lines[lines.length - 1]; // Get last line (should be JSON)
          const result = JSON.parse(jsonLine);
          
          if (result.error) {
            reject(new Error(result.error));
          } else {
            resolve(result);
          }
        } catch (e) {
          reject(new Error(`Failed to parse Python output: ${e.message}`));
        }
      } else {
        reject(new Error(`Python script failed with code ${code}: ${errorOutput}`));
      }
    });
    
    python.on('error', (err) => {
      reject(new Error(`Python execution failed: ${err.message}`));
    });
  });
}

// Intelligent fallback recommendation (when Python fails)
function generateIntelligentFallback(inputData) {
  const cgpa = parseFloat(inputData.cgpa) || 3;
  const programming = parseInt(inputData.programming) || 0;
  const interests = {
    gaming: parseInt(inputData.gameDevelopment) || 1,
    web: parseInt(inputData.webDevelopment) || 1,
    ai: parseInt(inputData.artificialIntelligence) || 1,
    database: parseInt(inputData.databaseSystem) || 1,
    testing: parseInt(inputData.softwareValidation) || 1
  };
  const difficulty = parseInt(inputData.difficulty) || 2;
  const learningStyle = parseInt(inputData.learningStyle) || 1;
  
  // Smart scoring algorithm (based on your MATLAB logic)
  const baseScore = (cgpa / 5.0) * 2 + (programming / 5.0) * 1.5;
  
  const scores = {
    'Gaming': baseScore + (interests.gaming / 5.0) * 2,
    'Web Development': baseScore + (interests.web / 5.0) * 2,
    'Fuzzy Logic': baseScore + (interests.ai / 5.0) * 2,
    'Database Design': baseScore + (interests.database / 5.0) * 2,
    'Software Validation & Verification': baseScore + (interests.testing / 5.0) * 2
  };
  
  // Apply difficulty adjustments
  if (difficulty == 1) {
    scores['Web Development'] *= 1.2;
    scores['Database Design'] *= 1.2;
  } else if (difficulty == 2) {
    scores['Gaming'] *= 1.1;
    scores['Software Validation & Verification'] *= 1.1;
  } else if (difficulty == 3) {
    scores['Fuzzy Logic'] *= 1.2;
  }
  
  // Apply learning style adjustments
  if (learningStyle == 1) {
    scores['Gaming'] *= 1.1;
    scores['Fuzzy Logic'] *= 1.1;
    scores['Database Design'] *= 1.1;
  } else if (learningStyle == 2) {
    scores['Gaming'] *= 1.15;
    scores['Web Development'] *= 1.15;
    scores['Database Design'] *= 1.15;
  } else if (learningStyle == 3) {
    scores['Web Development'] *= 1.1;
    scores['Software Validation & Verification'] *= 1.1;
  } else if (learningStyle == 4) {
    scores['Fuzzy Logic'] *= 1.05;
    scores['Software Validation & Verification'] *= 1.05;
  }
  
  // Add small random factor for variation
  for (let course in scores) {
    scores[course] += Math.random() * 0.1;
  }
  
  const sortedEntries = Object.entries(scores).sort(([,a], [,b]) => b - a);
  
  return {
    firstRecommendedCourse: sortedEntries[0][0],
    alternativeRecommendedCourse: sortedEntries[1][0],
    firstConfidence: Math.round(sortedEntries[0][1] * 100) / 100,
    secondConfidence: Math.round(sortedEntries[1][1] * 100) / 100,
    Confidence_Expert: Math.round(sortedEntries[0][1] * 100) / 100,
    Confidence_Tree: Math.round(sortedEntries[0][1] * 100) / 100,
    probability_Gaming: Math.round(scores['Gaming'] * 100) / 100,
    probability_WebDevelopment: Math.round(scores['Web Development'] * 100) / 100,
    probability_FuzzyLogic: Math.round(scores['Fuzzy Logic'] * 100) / 100,
    probability_DatabaseDesign: Math.round(scores['Database Design'] * 100) / 100,
    probability_SoftwareValidation_Verification: Math.round(scores['Software Validation & Verification'] * 100) / 100,
    expertRecommendation: sortedEntries[0][0],
    treeRecommendation: sortedEntries[0][0],
    finalRecommendation: sortedEntries[0][0],
    fisOutput: Math.round((sortedEntries[0][1] + sortedEntries[1][1]) / 2 * 100) / 100,
    method: 'Intelligent Fallback Algorithm',
    note: 'Generated using backup system for reliability',
    timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19)
  };
}

// Serve static pages
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

app.get('/input.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'input.html'));
});

app.get('/result.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'result.html'));
});

app.get('/contact.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'contact.html'));
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Course Recommendation Server running on port ${PORT}`);
  console.log(`🌐 Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`🔗 Health check: http://localhost:${PORT}/api/health`);
  console.log(`📁 Frontend served from: ${path.join(__dirname, 'frontend')}`);
});