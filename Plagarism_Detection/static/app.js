/**
 * VeriText AI - Frontend Client Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Inputs
    const textDocA = document.getElementById('textDocA');
    const textDocB = document.getElementById('textDocB');
    const statsDocA = document.getElementById('statsDocA');
    const statsDocB = document.getElementById('statsDocB');
    const fileInputA = document.getElementById('fileInputA');
    const fileInputB = document.getElementById('fileInputB');
    const clearBtnA = document.getElementById('clearBtnA');
    const clearBtnB = document.getElementById('clearBtnB');
    const swapDocsBtn = document.getElementById('swapDocsBtn');
    const presetSelect = document.getElementById('presetSelect');
    const modelSelect = document.getElementById('modelSelect');
    const thresholdRange = document.getElementById('thresholdRange');
    const thresholdVal = document.getElementById('thresholdVal');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeSpinner = document.getElementById('analyzeSpinner');
    const resetBtn = document.getElementById('resetBtn');
    const themeToggle = document.getElementById('themeToggle');

    // DOM Elements - Results
    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const resultsSection = document.getElementById('resultsSection');
    const verdictBanner = document.getElementById('verdictBanner');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictDesc = document.getElementById('verdictDesc');
    const verdictBadge = document.getElementById('verdictBadge');

    // DOM Elements - Metrics
    const metricOverallScore = document.getElementById('metricOverallScore');
    const scoreRingFill = document.getElementById('scoreRingFill');
    const metricMatchedRatio = document.getElementById('metricMatchedRatio');
    const progressMatchedRatio = document.getElementById('progressMatchedRatio');
    const metricSentenceCounts = document.getElementById('metricSentenceCounts');
    const metricAlignmentsCount = document.getElementById('metricAlignmentsCount');
    const metricThresholdVal = document.getElementById('metricThresholdVal');
    const metricTopTier = document.getElementById('metricTopTier');
    const metricTopTierDesc = document.getElementById('metricTopTierDesc');

    // DOM Elements - Tiers
    const countVerbatim = document.getElementById('countVerbatim');
    const countLight = document.getElementById('countLight');
    const countHeavy = document.getElementById('countHeavy');
    const countStructural = document.getElementById('countStructural');
    const tierCards = document.querySelectorAll('.tier-card');

    // DOM Elements - Highlighters & Table
    const highlighterDocA = document.getElementById('highlighterDocA');
    const highlighterDocB = document.getElementById('highlighterDocB');
    const clearHighlightBtn = document.getElementById('clearHighlightBtn');
    const tableBody = document.getElementById('tableBody');
    const tableSearch = document.getElementById('tableSearch');
    const exportJsonBtn = document.getElementById('exportJsonBtn');
    const printReportBtn = document.getElementById('printReportBtn');

    // Global State
    let analysisData = null;
    let presetsData = {};
    let activeTierFilters = new Set(['verbatim', 'light_paraphrase', 'heavy_paraphrase', 'structural_rewrite']);

    // --- 1. Theme Management ---
    const savedTheme = localStorage.getItem('veritext_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('veritext_theme', newTheme);
        updateThemeIcon(newTheme);
    });

    function updateThemeIcon(theme) {
        themeToggle.innerHTML = theme === 'dark' ? '<i class="fa-solid fa-sun"></i>' : '<i class="fa-solid fa-moon"></i>';
    }

    // --- 2. Word & Character Counters ---
    function updateTextStats(textarea, statsEl) {
        const text = textarea.value.trim();
        const words = text ? text.split(/\s+/).length : 0;
        const chars = text.length;
        statsEl.innerHTML = `<i class="fa-solid fa-font"></i> ${words} words | ${chars} chars`;
    }

    textDocA.addEventListener('input', () => updateTextStats(textDocA, statsDocA));
    textDocB.addEventListener('input', () => updateTextStats(textDocB, statsDocB));

    // Clear & Swap Actions
    clearBtnA.addEventListener('click', () => { textDocA.value = ''; updateTextStats(textDocA, statsDocA); });
    clearBtnB.addEventListener('click', () => { textDocB.value = ''; updateTextStats(textDocB, statsDocB); });
    
    swapDocsBtn.addEventListener('click', () => {
        const temp = textDocA.value;
        textDocA.value = textDocB.value;
        textDocB.value = temp;
        updateTextStats(textDocA, statsDocA);
        updateTextStats(textDocB, statsDocB);
    });

    // Sensitivity Threshold Slider
    thresholdRange.addEventListener('input', () => {
        thresholdVal.textContent = parseFloat(thresholdRange.value).toFixed(2);
    });

    // Model select change re-runs analysis if text is present
    if (modelSelect) {
        modelSelect.addEventListener('change', () => {
            if (textDocA.value.trim() && textDocB.value.trim()) {
                runAnalysis();
            }
        });
    }

    // --- 3. Presets & File Uploads ---
    fetchPresets();

    async function fetchPresets() {
        try {
            const res = await fetch('/api/presets');
            if (res.ok) {
                presetsData = await res.json();
            }
        } catch (err) {
            console.error('Failed to fetch presets:', err);
        }
    }

    presetSelect.addEventListener('change', () => {
        const key = presetSelect.value;
        if (presetsData[key]) {
            textDocA.value = presetsData[key].doc_a;
            textDocB.value = presetsData[key].doc_b;
            updateTextStats(textDocA, statsDocA);
            updateTextStats(textDocB, statsDocB);
            runAnalysis();
        }
    });

    function setupFileInput(inputEl, textareaEl, statsEl) {
        inputEl.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (evt) => {
                    textareaEl.value = evt.target.result;
                    updateTextStats(textareaEl, statsEl);
                };
                reader.readAsText(file);
            }
        });
    }

    setupFileInput(fileInputA, textDocA, statsDocA);
    setupFileInput(fileInputB, textDocB, statsDocB);

    // --- 4. Main Analysis Request ---
    analyzeBtn.addEventListener('click', runAnalysis);
    resetBtn.addEventListener('click', resetAll);

    async function runAnalysis() {
        const docA = textDocA.value.trim();
        const docB = textDocB.value.trim();
        const selectedModel = modelSelect ? modelSelect.value : 'sbert';

        if (!docA || !docB) {
            alert('Please provide text for both Document A and Document B before analyzing.');
            return;
        }

        // Show UI Loading State
        analyzeBtn.disabled = true;
        analyzeSpinner.style.display = 'inline-block';

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    doc_a: docA,
                    doc_b: docB,
                    threshold: parseFloat(thresholdRange.value),
                    model_type: selectedModel
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error occurred during analysis.');
            }

            analysisData = await response.json();
            renderResults(analysisData);

        } catch (err) {
            alert('Analysis Error: ' + err.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeSpinner.style.display = 'none';
        }
    }

    function resetAll() {
        textDocA.value = '';
        textDocB.value = '';
        updateTextStats(textDocA, statsDocA);
        updateTextStats(textDocB, statsDocB);
        presetSelect.value = '';
        resultsSection.classList.add('hidden');
        resultsPlaceholder.classList.remove('hidden');
        analysisData = null;
    }

    // --- 5. Render Results Dashboard ---
    function renderResults(data) {
        resultsPlaceholder.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        const summary = data.summary;

        // A. Verdict Banner
        verdictBanner.className = 'verdict-banner ' + (summary.flagged ? 'flagged' : 'cleared');
        if (summary.flagged) {
            verdictTitle.textContent = 'FLAGGED — High Paraphrase / Duplicate Risk';
            verdictDesc.textContent = `Overall semantic similarity of ${(summary.overall_score * 100).toFixed(1)}% across ${summary.matched_count} matched passages.`;
            verdictBadge.textContent = 'FLAGGED';
        } else {
            verdictTitle.textContent = 'CLEARED — Low Paraphrase Risk';
            verdictDesc.textContent = `Minimal duplicate content detected (${(summary.overall_score * 100).toFixed(1)}% semantic similarity).`;
            verdictBadge.textContent = 'CLEARED';
        }

        // B. Metrics Grid
        const pctScore = Math.round(summary.overall_score * 100);
        metricOverallScore.textContent = pctScore + '%';
        // Ring offset math (r=42 -> circumference ~ 264)
        const offset = 264 - (264 * pctScore / 100);
        scoreRingFill.style.strokeDashoffset = offset;

        const pctMatched = Math.round(summary.matched_sentence_ratio * 100);
        metricMatchedRatio.textContent = pctMatched + '%';
        progressMatchedRatio.style.width = pctMatched + '%';
        metricSentenceCounts.textContent = `${summary.matched_count} of ${summary.total_sentences_a} sentences matched`;

        metricAlignmentsCount.textContent = summary.matched_count;
        metricThresholdVal.textContent = parseFloat(thresholdRange.value).toFixed(2);

        // Highest Intensity Tier
        const counts = summary.tier_counts || {};
        let topTier = 'None';
        let topTierDesc = 'No matches above threshold';

        if (counts.verbatim > 0) {
            topTier = 'Verbatim Copy';
            topTierDesc = `${counts.verbatim} sentence(s) verbatim`;
            metricTopTier.className = 'top-tier-badge tier-badge-verbatim';
        } else if (counts.light_paraphrase > 0) {
            topTier = 'Light Paraphrase';
            topTierDesc = `${counts.light_paraphrase} sentence(s) lightly rephrased`;
            metricTopTier.className = 'top-tier-badge tier-badge-light';
        } else if (counts.heavy_paraphrase > 0) {
            topTier = 'Heavy Paraphrase';
            topTierDesc = `${counts.heavy_paraphrase} sentence(s) heavily rewritten`;
            metricTopTier.className = 'top-tier-badge tier-badge-heavy';
        } else if (counts.structural_rewrite > 0) {
            topTier = 'Structural Rewrite';
            topTierDesc = `${counts.structural_rewrite} sentence(s) structural alignment`;
            metricTopTier.className = 'top-tier-badge tier-badge-structural';
        } else {
            metricTopTier.className = 'top-tier-badge';
        }

        metricTopTier.textContent = topTier;
        metricTopTierDesc.textContent = topTierDesc;

        // C. Tier Breakdown Counts
        countVerbatim.textContent = counts.verbatim || 0;
        countLight.textContent = counts.light_paraphrase || 0;
        countHeavy.textContent = counts.heavy_paraphrase || 0;
        countStructural.textContent = counts.structural_rewrite || 0;

        // D. Side-by-Side Highlighters
        renderHighlighters(data.doc_a.sentences, data.doc_b.sentences);

        // E. Alignment Table
        renderTable(data.matches);
    }

    // --- 6. Render Highlighters ---
    function renderHighlighters(sentsA, sentsB) {
        highlighterDocA.innerHTML = '';
        highlighterDocB.innerHTML = '';

        sentsA.forEach((s) => {
            const span = document.createElement('span');
            span.className = 'sent-span ' + (s.is_matched ? `matched tier-${s.tier}-hl` : '');
            span.textContent = s.text + ' ';
            if (s.is_matched) {
                span.dataset.matchId = s.match_id;
                span.dataset.pairedIndex = s.paired_index;
                span.dataset.tier = s.tier;
                span.dataset.doc = 'A';
                span.title = `Tier: ${s.tier.replace('_', ' ').toUpperCase()} | Sem: ${(s.semantic_similarity * 100).toFixed(1)}%`;
                span.addEventListener('click', () => handleSentenceClick(s.match_id, s.paired_index, 'B'));
            }
            highlighterDocA.appendChild(span);
        });

        sentsB.forEach((s) => {
            const span = document.createElement('span');
            span.className = 'sent-span ' + (s.is_matched ? `matched tier-${s.tier}-hl` : '');
            span.textContent = s.text + ' ';
            if (s.is_matched) {
                span.dataset.matchId = s.match_id;
                span.dataset.pairedIndex = s.paired_index;
                span.dataset.tier = s.tier;
                span.dataset.doc = 'B';
                span.title = `Tier: ${s.tier.replace('_', ' ').toUpperCase()} | Sem: ${(s.semantic_similarity * 100).toFixed(1)}%`;
                span.addEventListener('click', () => handleSentenceClick(s.match_id, s.paired_index, 'A'));
            }
            highlighterDocB.appendChild(span);
        });
    }

    function handleSentenceClick(matchId, pairedIdx, targetDoc) {
        clearFocusHighlights();
        
        const sourceSpans = document.querySelectorAll(`.sent-span[data-match-id="${matchId}"]`);
        sourceSpans.forEach(s => s.classList.add('active-focus'));

        const targetContainer = targetDoc === 'B' ? highlighterDocB : highlighterDocA;
        const targetSpan = targetContainer.querySelector(`.sent-span[data-match-id="${matchId}"]`);

        if (targetSpan) {
            targetSpan.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    function clearFocusHighlights() {
        document.querySelectorAll('.sent-span.active-focus').forEach(s => s.classList.remove('active-focus'));
    }

    clearHighlightBtn.addEventListener('click', clearFocusHighlights);

    // --- 7. Tier Filter Chips ---
    tierCards.forEach(card => {
        card.addEventListener('click', () => {
            const tier = card.dataset.tier;
            if (activeTierFilters.has(tier)) {
                activeTierFilters.delete(tier);
                card.classList.remove('active');
            } else {
                activeTierFilters.add(tier);
                card.classList.add('active');
            }
            applyTierFilters();
        });
    });

    function applyTierFilters() {
        if (!analysisData) return;

        document.querySelectorAll('.sent-span.matched').forEach(span => {
            const tier = span.dataset.tier;
            if (activeTierFilters.has(tier)) {
                span.style.opacity = '1';
                span.style.pointerEvents = 'auto';
            } else {
                span.style.opacity = '0.35';
                span.style.pointerEvents = 'none';
            }
        });

        renderTable(analysisData.matches);
    }

    // --- 8. Alignment Table Rendering & Search ---
    function renderTable(matches) {
        tableBody.innerHTML = '';
        const search = tableSearch.value.toLowerCase().trim();

        const filtered = matches.filter(m => {
            const matchesTier = activeTierFilters.has(m.tier);
            const matchesSearch = !search || 
                m.sentence_a.toLowerCase().includes(search) || 
                m.sentence_b.toLowerCase().includes(search) ||
                m.tier_label.toLowerCase().includes(search);
            return matchesTier && matchesSearch;
        });

        if (filtered.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 24px; color: var(--text-muted);">No matched passages found matching the current filters.</td></tr>`;
            return;
        }

        filtered.forEach(m => {
            const tr = document.createElement('tr');
            
            const badgeClass = m.tier === 'verbatim' ? 'tier-badge-verbatim' :
                               m.tier === 'light_paraphrase' ? 'tier-badge-light' :
                               m.tier === 'heavy_paraphrase' ? 'tier-badge-heavy' : 'tier-badge-structural';

            tr.innerHTML = `
                <td><strong>#${m.match_id}</strong></td>
                <td>${escapeHtml(m.sentence_a)}</td>
                <td>${escapeHtml(m.sentence_b)}</td>
                <td><strong>${(m.semantic_similarity * 100).toFixed(1)}%</strong></td>
                <td>${(m.lexical_similarity * 100).toFixed(1)}%</td>
                <td><span class="tier-badge-pill ${badgeClass}">${m.tier_label}</span></td>
            `;

            tr.style.cursor = 'pointer';
            tr.addEventListener('click', () => handleSentenceClick(m.match_id, m.index_b, 'B'));
            tableBody.appendChild(tr);
        });
    }

    tableSearch.addEventListener('input', () => {
        if (analysisData) renderTable(analysisData.matches);
    });

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // --- 9. Export Actions ---
    exportJsonBtn.addEventListener('click', () => {
        if (!analysisData) return;
        const blob = new Blob([JSON.stringify(analysisData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'veritext_analysis_report.json';
        a.click();
        URL.revokeObjectURL(url);
    });

    printReportBtn.addEventListener('click', () => {
        window.print();
    });
});
