
from flask import Flask, render_template_string, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ==========================================
# 1. LOAD TRAINED ML MODELS
# ==========================================

MODEL_FOLDER = "models"

knn_model = joblib.load(
    os.path.join(MODEL_FOLDER, "knn.pkl")
)

decision_tree_model = joblib.load(
    os.path.join(MODEL_FOLDER, "decision_tree.pkl")
)

random_forest_model = joblib.load(
    os.path.join(MODEL_FOLDER, "random_forest.pkl")
)

label_encoder = joblib.load(
    os.path.join(MODEL_FOLDER, "label_encoder.pkl")
)

best_model_name = joblib.load(
    os.path.join(MODEL_FOLDER, "best_model_name.pkl")
)

# ==========================================
# 2. LOAD JOB SKILLS DATASET
# ==========================================

data = pd.read_csv("dataset/job_skills.csv")

job_roles = {}

for _, row in data.iterrows():

    role = row["job_role"]

    skills = row["skills"].split()

    if role not in job_roles:
        job_roles[role] = set()

    job_roles[role].update(skills)

# ==========================================
# 3. MACHINE LEARNING PREDICTION
# ==========================================

def predict_job(student_skills):

    student_text = " ".join(student_skills)

    models = {
        "KNN": knn_model,
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model
    }

    results = []

    for name, model in models.items():

        # Predict job role
        prediction = model.predict([student_text])[0]

        # Convert encoded number into job name
        job_name = label_encoder.inverse_transform(
            [prediction]
        )[0]

        # Get model confidence if available
        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                [student_text]
            )[0]

            confidence = round(
                max(probabilities) * 100, 2
            )

        results.append({
            "name": name,
            "job": job_name,
            "confidence": confidence
        })

    # Get best model prediction
    best_result = next(
        item for item in results
        if item["name"] == best_model_name
    )

    recommended_job = best_result["job"]

    # Find matched skills
    required_skills = job_roles.get(
        recommended_job, set()
    )

    student_skill_set = {
        skill.lower() for skill in student_skills
    }

    matched_skills = [
        skill for skill in sorted(required_skills)
        if skill.lower() in student_skill_set
    ]

    missing_skills = [
        skill for skill in sorted(required_skills)
        if skill.lower() not in student_skill_set
    ]

    # Calculate skill coverage percentage
    if required_skills:

        match_percentage = round(
            len(matched_skills) /
            len(required_skills) * 100, 2
        )

    else:
        match_percentage = 0

    return {
        "results": results,
        "best_model": best_model_name,
        "recommended_job": recommended_job,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "match_percentage": match_percentage
    }


# ==========================================
# 4. WEBSITE HTML
# ==========================================

HTML = """
<!DOCTYPE html>
<html>
<head>

    <title>SkillMatch AI</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <link rel="stylesheet"
          href="{{ url_for('static', filename='style.css') }}">

</head>

<body>

<header>

    <h1>SkillMatch AI</h1>

    <p>Student Skill Analysis and Job Recommendation System</p>

</header>

<div class="container">

    <div class="card">

        <h2>Analyze Your Skills</h2>

        <p>
            Enter your technical skills separated by commas.
        </p>

        <form method="POST">

            <textarea
                name="skills"
                placeholder="Example: Python, Java, HTML, CSS, SQL"
                required
            >{{ entered_skills }}</textarea>

            <button type="submit">
                Analyze My Skills
            </button>

        </form>

    </div>

    {% if result %}

    <div class="card">

        <h2>Job Recommendation</h2>

        <h3>{{ result.recommended_job }}</h3>

        <p>Best Performing Algorithm:</p>

        <h3 class="percentage">
            {{ result.best_model }}
        </h3>

        <p>Skill Coverage:</p>

        <div class="percentage">
            {{ result.match_percentage }}%
        </div>

        <p>
            This percentage represents the proportion
            of listed job skills that you currently have.
        </p>

    </div>


    <div class="card">

        <h2>ML Algorithm Comparison</h2>

        {% for item in result.results %}

            <div class="job">

                <h3>{{ item.name }}</h3>

                <p>
                    Predicted Job: <strong>{{ item.job }}</strong>
                </p>

                {% if item.confidence is not none %}

                    <p>
                        Model Confidence:
                        <strong>{{ item.confidence }}%</strong>
                    </p>

                {% endif %}

            </div>

        {% endfor %}

    </div>


    <div class="card">

        <h2 class="matched">Your Matched Skills</h2>

        {% if result.matched_skills %}

            {% for skill in result.matched_skills %}

                <span class="skill">{{ skill }}</span>

            {% endfor %}

        {% else %}

            <p>No exact skill matches found.</p>

        {% endif %}

    </div>


    <div class="card">

        <h2 class="missing">Skills You Need to Learn</h2>

        {% if result.missing_skills %}

            {% for skill in result.missing_skills %}

                <span class="skill">{{ skill }}</span>

            {% endfor %}

        {% else %}

            <p>You have all listed required skills!</p>

        {% endif %}

    </div>

    {% endif %}

</div>

<footer>

    <p>SkillMatch AI | Machine Learning Project</p>

</footer>

</body>
</html>
"""


# ==========================================
# 5. HOME PAGE ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    entered_skills = ""

    if request.method == "POST":

        entered_skills = request.form.get(
            "skills", ""
        ).strip()

        student_skills = [
            skill.strip()
            for skill in entered_skills.split(",")
            if skill.strip()
        ]

        if student_skills:

            result = predict_job(student_skills)

    return render_template_string(
        HTML,
        result=result,
        entered_skills=entered_skills
    )


# ==========================================
# 6. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)
