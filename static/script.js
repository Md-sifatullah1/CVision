const form = document.querySelector('#analysis-form');
const statusEl = document.querySelector('#status');
const button = document.querySelector('#submit');
const results = document.querySelector('#results');
const API_BASE_URL = 'http://localhost:8000';

function item(parent, title, detail, isGap = false) {
  const box = document.createElement('div'); box.className = 'item';
  const strong = document.createElement('strong'); strong.textContent = title;
  const small = document.createElement('small'); small.textContent = detail || '';
  box.append(strong, small); parent.append(box);
}
function render(data) {
  for (const id of ['matches', 'gaps', 'questions', 'topics']) document.getElementById(id).replaceChildren();
  data.matching_skills.forEach(m => item(document.querySelector('#matches'), m.skill, `CV evidence: ${m.cv_evidence}`));
  data.missing_requirements.forEach(g => item(document.querySelector('#gaps'), g.skill, `Job requirement: ${g.requirement}`));
  data.interview_questions.forEach(q => item(document.querySelector('#questions'), q.question, `${q.category.replace('_', ' ')} · ${q.related_skill}`));
  data.study_topics.forEach(t => item(document.querySelector('#topics'), t.topic, t.reason));
  for (const [id, items] of [['matches', data.matching_skills], ['gaps', data.missing_requirements], ['questions', data.interview_questions], ['topics', data.study_topics]]) {
    if (!items.length) item(document.getElementById(id), 'None identified', 'This baseline recognizes only skills in its dictionary.');
  }
  document.querySelector('#disclaimer').textContent = data.disclaimer;
  document.querySelector('#method').textContent = data.method;
  results.hidden = false; results.scrollIntoView({behavior:'smooth'});
}
form.addEventListener('submit', async event => {
  event.preventDefault();

  const file = document.querySelector('#cv').files[0];

  if (!file) {
    statusEl.textContent = 'Please select a CV file.';
    return;
  }

  if (file.size > 5 * 1024 * 1024) {
    statusEl.textContent = 'Your CV must be 5 MB or smaller.';
    return;
  }

  statusEl.textContent = ''; results.hidden = true;
  button.disabled = true; button.textContent = 'Analyzing...';
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/analyze-file`, {
      method: 'POST',
      body: new FormData(form)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid request. Please check your input.');
    render(data);
  } catch (error) { statusEl.textContent = error.message || 'Analysis failed. Please try again.'; }
  finally { button.disabled = false; button.textContent = 'Analyze my CV →'; }
});
