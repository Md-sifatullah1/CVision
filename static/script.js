
const API_BASE_URL = '';

const form = document.querySelector('#analysis-form');
const statusEl = document.querySelector('#status');
const button = document.querySelector('#submit');
const results = document.querySelector('#results');


// Create a result item
function item(parent, title, detail) {
  const box = document.createElement('div');
  box.className = 'item';

  const strong = document.createElement('strong');
  strong.textContent = title;

  const small = document.createElement('small');
  small.textContent = detail || '';

  box.append(strong, small);
  parent.append(box);
}


// Display CV analysis results
function render(data) {
  const ids = ['matches', 'gaps', 'questions', 'topics'];

  for (const id of ids) {
    document.getElementById(id).replaceChildren();
  }

  // Matching skills
  data.matching_skills.forEach(match => {
    item(
        document.querySelector('#matches'),
        match.skill,
        `CV evidence: ${match.cv_evidence}`
    );
  });

  // Missing requirements
  data.missing_requirements.forEach(gap => {
    item(
        document.querySelector('#gaps'),
        gap.skill,
        `Job requirement: ${gap.requirement}`
    );
  });

  // Interview questions
  data.interview_questions.forEach(question => {
    item(
        document.querySelector('#questions'),
        question.question,
        `${question.category.replaceAll('_', ' ')} · ${question.related_skill}`
    );
  });

  // Suggested study topics
  data.study_topics.forEach(topic => {
    item(
        document.querySelector('#topics'),
        topic.topic,
        topic.reason
    );
  });

  // Handle empty result sections
  const groups = [
    ['matches', data.matching_skills],
    ['gaps', data.missing_requirements],
    ['questions', data.interview_questions],
    ['topics', data.study_topics],
  ];

  for (const [id, items] of groups) {
    if (!items.length) {
      item(
          document.getElementById(id),
          'None identified',
          'No relevant items were identified.'
      );
    }
  }

  // Analysis metadata
  document.querySelector('#disclaimer').textContent =
      data.disclaimer;


  // Show results
  results.hidden = false;
  results.scrollIntoView({ behavior: 'smooth' });
}


// Handle CV upload and analysis
form.addEventListener('submit', async event => {
  event.preventDefault();

  const file = document.querySelector('#cv').files[0];
  const jobDescription =
      document.querySelector('#jd').value.trim();

  // Validate CV
  if (!file) {
    statusEl.textContent = 'Please select a CV file.';
    return;
  }

  const allowedExtensions = ['pdf', 'docx'];
  const extension = file.name.split('.').pop().toLowerCase();

  if (!allowedExtensions.includes(extension)) {
    statusEl.textContent =
        'Please upload a PDF or DOCX file.';
    return;
  }

  if (file.size > 5 * 1024 * 1024) {
    statusEl.textContent =
        'Your CV must be 5 MB or smaller.';
    return;
  }

  if (!jobDescription) {
    statusEl.textContent =
        'Please enter a job description.';
    return;
  }

  // Reset previous results
  statusEl.textContent = '';
  results.hidden = true;

  button.disabled = true;
  button.textContent = 'Analyzing...';

  try {
    const response = await fetch(
        `${API_BASE_URL}/api/v1/analyze-file`,
        {
          method: 'POST',
          body: new FormData(form),
        }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
          typeof data.detail === 'string'
              ? data.detail
              : 'Invalid request. Please check your input.'
      );
    }

    render(data);

  } catch (error) {
    statusEl.textContent =
        error.message || 'Analysis failed. Please try again.';

  } finally {
    button.disabled = false;
    button.textContent = 'Analyze my CV →';
  }
});
