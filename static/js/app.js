// Highlight active section pill as user scrolls
const sections = ['demographics', 'clinical', 'medications', 'socioeconomic', 'lifestyle', 'comorbidities'];
const pills = document.querySelectorAll('.step-pill');

function updateActivePill() {
  let current = sections[0];
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el && window.scrollY >= el.offsetTop - 120) current = id;
  });
  pills.forEach(pill => {
    const href = pill.getAttribute('href').replace('#', '');
    const active = href === current;
    pill.classList.toggle('border-primary-600', active);
    pill.classList.toggle('text-primary-600', active);
    pill.classList.toggle('bg-primary-50', active);
    pill.classList.toggle('border-slate-300', !active);
    pill.classList.toggle('text-slate-500', !active);
  });
}

window.addEventListener('scroll', updateActivePill, { passive: true });
updateActivePill();

// Basic validation + loading state on submit
const form = document.getElementById('main-form');
const submitBtn = document.getElementById('submit-btn');
if (form && submitBtn) {
  form.addEventListener('submit', (e) => {
    const required = form.querySelectorAll('[required]');
    let valid = true;
    required.forEach(el => {
      if (el.type === 'radio') return;
      if (!el.value.trim()) {
        el.classList.add('ring-2', 'ring-red-400');
        valid = false;
      } else {
        el.classList.remove('ring-2', 'ring-red-400');
      }
    });
    if (!valid) {
      e.preventDefault();
      const first = form.querySelector('.ring-red-400');
      if (first) first.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }
    submitBtn.disabled = true;
    submitBtn.textContent = 'Predicting…';
    submitBtn.classList.add('opacity-75', 'cursor-not-allowed');
  });

  // Clear red ring on input
  form.querySelectorAll('input, select').forEach(el => {
    el.addEventListener('input', () => el.classList.remove('ring-2', 'ring-red-400'));
  });
}
