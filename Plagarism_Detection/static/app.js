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
            tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding: 24px; color: var(--text-muted);">No matched passages found matching the current filters.</td></tr>`;
            return;
        }

        filtered.forEach(m => {
            const tr = document.createElement('tr');
            
            const badgeClass = m.tier === 'verbatim' ? 'tier-badge-verbatim' :
                               m.tier === 'light_paraphrase' ? 'tier-badge-light' :
                               m.tier === 'heavy_paraphrase' ? 'tier-badge-heavy' : 'tier-badge-structural';

            const ceDisplay = (m.cross_encoder_score !== null && m.cross_encoder_score !== undefined)
                ? `<span class="score-chip score-chip-ce">${(m.cross_encoder_score * 100).toFixed(1)}%</span>`
                : `<span style="color: var(--text-muted); font-size: 0.8rem;">—</span>`;

            tr.innerHTML = `
                <td><strong>#${m.match_id}</strong></td>
                <td>${escapeHtml(m.sentence_a)}</td>
                <td>${escapeHtml(m.sentence_b)}</td>
                <td><span class="score-chip score-chip-bi">${(m.semantic_similarity * 100).toFixed(1)}%</span></td>
                <td>${ceDisplay}</td>
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

    // --- 10. Navigation Tabs (Pairwise, FAISS, and AI Detection) ---
    const tabBtnPairwise = document.getElementById('tabBtnPairwise');
    const tabBtnFaiss = document.getElementById('tabBtnFaiss');
    const tabBtnAIDetect = document.getElementById('tabBtnAIDetect');
    const pairwiseView = document.getElementById('pairwiseView');
    const faissView = document.getElementById('faissView');
    const aiDetectView = document.getElementById('aiDetectView');

    const tabViews = [
        { key: 'pairwise', button: tabBtnPairwise, view: pairwiseView },
        { key: 'faiss', button: tabBtnFaiss, view: faissView },
        { key: 'aiDetect', button: tabBtnAIDetect, view: aiDetectView },
    ];

    function switchTab(target) {
        tabViews.forEach(({ key, button, view }) => {
            if (button) button.classList.toggle('active', key === target);
            if (view) view.style.display = key === target ? 'block' : 'none';
        });
    }

    if (tabBtnPairwise) tabBtnPairwise.addEventListener('click', () => switchTab('pairwise'));
    if (tabBtnFaiss) tabBtnFaiss.addEventListener('click', () => switchTab('faiss'));
    if (tabBtnAIDetect) tabBtnAIDetect.addEventListener('click', () => switchTab('aiDetect'));

    // --- 11. FAISS Corpus Search & Re-ranking Logic ---
    const faissQueryText = document.getElementById('faissQueryText');
    const faissTopK = document.getElementById('faissTopK');
    const faissRerankToggle = document.getElementById('faissRerankToggle');
    const faissSearchBtn = document.getElementById('faissSearchBtn');
    const faissSpinner = document.getElementById('faissSpinner');
    const faissResultsSection = document.getElementById('faissResultsSection');
    const faissCandidatesList = document.getElementById('faissCandidatesList');
    const faissResultsSubtitle = document.getElementById('faissResultsSubtitle');
    const faissCorpusStatsBadge = document.getElementById('faissCorpusStatsBadge');
    const browseCorpusBtn = document.getElementById('browseCorpusBtn');
    const corpusBrowseSection = document.getElementById('corpusBrowseSection');
    const closeCorpusBrowseBtn = document.getElementById('closeCorpusBrowseBtn');
    const corpusGrid = document.getElementById('corpusGrid');

    // Sample query buttons
    document.querySelectorAll('.sample-query-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            faissQueryText.value = btn.dataset.query;
        });
    });

    if (faissSearchBtn) {
        faissSearchBtn.addEventListener('click', async () => {
            const query = faissQueryText.value.trim();
            if (!query) {
                alert('Please enter a query text or click one of the sample prompt buttons.');
                return;
            }

            faissSearchBtn.disabled = true;
            if (faissSpinner) faissSpinner.style.display = 'inline-block';

            try {
                const topK = parseInt(faissTopK.value, 10) || 5;
                const rerank = faissRerankToggle ? faissRerankToggle.checked : true;

                const response = await fetch('/api/faiss/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        top_k: topK,
                        rerank: rerank
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.error || 'FAISS search error');
                }

                const result = await response.json();
                renderFaissResults(query, result);
            } catch (err) {
                alert('Search Error: ' + err.message);
            } finally {
                faissSearchBtn.disabled = false;
                if (faissSpinner) faissSpinner.style.display = 'none';
            }
        });
    }

    function renderFaissResults(query, result) {
        faissResultsSection.classList.remove('hidden');
        faissCandidatesList.innerHTML = '';

        if (faissCorpusStatsBadge) {
            faissCorpusStatsBadge.innerHTML = `<i class="fa-solid fa-database"></i> ${result.total_indexed} Documents Indexed`;
        }

        if (faissResultsSubtitle) {
            faissResultsSubtitle.textContent = result.reranked_with_cross_encoder
                ? `Retrieved ${result.candidates.length} candidates with FAISS vector search + Cross-Encoder 2-stage re-ranking`
                : `Retrieved ${result.candidates.length} candidates with fast FAISS L2-normalized vector similarity`;
        }

        if (!result.candidates || result.candidates.length === 0) {
            faissCandidatesList.innerHTML = '<p style="color:var(--text-muted); padding:20px; text-align:center;">No candidates found.</p>';
            return;
        }

        result.candidates.forEach((cand, idx) => {
            const card = document.createElement('div');
            card.className = 'candidate-card';

            const biScoreVal = cand.bi_encoder_similarity !== undefined ? cand.bi_encoder_similarity : (cand.bi_encoder_score || 0.0);
            const biScorePercent = (biScoreVal * 100).toFixed(1);
            const ceScoreDisplay = (cand.cross_encoder_score !== undefined && cand.cross_encoder_score !== null)
                ? `<span class="score-chip score-chip-ce" title="Cross-Encoder Re-ranker Score"><i class="fa-solid fa-crosshairs"></i> Cross-Encoder: ${(cand.cross_encoder_score * 100).toFixed(1)}%</span>`
                : '';

            const authorStr = cand.author || (cand.metadata ? cand.metadata.author : '');
            const titleStr = cand.title || (cand.metadata ? cand.metadata.title : '');
            const authorInfo = authorStr ? `${authorStr}${titleStr ? ' — ' + titleStr : ''}` : titleStr;

            card.innerHTML = `
                <div class="candidate-card-header">
                    <div class="candidate-title-info">
                        <span class="candidate-rank-badge">Rank #${cand.rank || (idx + 1)}</span>
                        <span class="candidate-id-badge">${escapeHtml(cand.doc_id)}</span>
                        ${authorInfo ? `<span style="font-size:0.85rem; color:var(--text-muted);">${escapeHtml(authorInfo)}</span>` : ''}
                    </div>
                    <div class="candidate-scores-group">
                        <span class="score-chip score-chip-bi" title="FAISS SBERT Cosine Similarity"><i class="fa-solid fa-vector-square"></i> FAISS Sim: ${biScorePercent}%</span>
                        ${ceScoreDisplay}
                    </div>
                </div>
                <div class="candidate-snippet">${escapeHtml(cand.text)}</div>
                <div class="candidate-card-actions">
                    <button type="button" class="btn btn-sm btn-primary compare-cand-btn">
                        <i class="fa-solid fa-code-compare"></i> Compare in Pairwise Studio
                    </button>
                </div>
            `;

            const compareBtn = card.querySelector('.compare-cand-btn');
            compareBtn.addEventListener('click', () => {
                textDocA.value = query;
                textDocB.value = cand.text;
                updateTextStats(textDocA, statsDocA);
                updateTextStats(textDocB, statsDocB);
                switchTab('pairwise');
                runAnalysis();
            });

            faissCandidatesList.appendChild(card);
        });

        faissResultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Browse Corpus Library
    if (browseCorpusBtn) {
        browseCorpusBtn.addEventListener('click', async () => {
            corpusBrowseSection.classList.remove('hidden');
            corpusGrid.innerHTML = '<p style="color:var(--text-muted); padding:20px;">Loading indexed documents...</p>';

            try {
                const res = await fetch('/api/corpus');
                if (res.ok) {
                    const data = await res.json();
                    renderCorpusLibrary(data.corpus || []);
                }
            } catch (err) {
                corpusGrid.innerHTML = `<p style="color:var(--text-muted); padding:20px;">Failed to load corpus: ${err.message}</p>`;
            }

            corpusBrowseSection.scrollIntoView({ behavior: 'smooth' });
        });
    }

    if (closeCorpusBrowseBtn) {
        closeCorpusBrowseBtn.addEventListener('click', () => {
            corpusBrowseSection.classList.add('hidden');
        });
    }

    function renderCorpusLibrary(corpus) {
        corpusGrid.innerHTML = '';
        corpus.forEach(item => {
            const card = document.createElement('div');
            card.className = 'corpus-card';
            card.innerHTML = `
                <div class="corpus-card-header">
                    <strong>${escapeHtml(item.doc_id)}</strong>
                    <span>${escapeHtml(item.author || '')}</span>
                </div>
                <div class="corpus-card-title">${escapeHtml(item.title || 'Untitled')}</div>
                <div class="corpus-card-text">${escapeHtml(item.text)}</div>
                <div style="margin-top:10px; display:flex; justify-content:flex-end;">
                    <button type="button" class="btn btn-sm btn-outline load-corpus-doc-btn">
                        <i class="fa-solid fa-upload"></i> Load as Query
                    </button>
                </div>
            `;

            card.querySelector('.load-corpus-doc-btn').addEventListener('click', () => {
                faissQueryText.value = item.text;
                window.scrollTo({ top: faissQueryText.offsetTop - 100, behavior: 'smooth' });
            });

            corpusGrid.appendChild(card);
        });
    }

    // --- 12. AI-Generated Text Detection UI ---
    const aiDetectText = document.getElementById('aiDetectText');
    const aiDetectBtn = document.getElementById('aiDetectBtn');
    const aiDetectSpinner = document.getElementById('aiDetectSpinner');
    const aiDetectStatus = document.getElementById('aiDetectStatus');
    const aiDetectResults = document.getElementById('aiDetectResults');
    const aiDetectLoadSample = document.getElementById('aiDetectLoadSample');
    const aiDetectClear = document.getElementById('aiDetectClear');
    const aiDetectCharCount = document.getElementById('aiDetectCharCount');
    const aiDetectSentenceList = document.getElementById('aiDetectSentenceList');

    const aiDetectSample = `Renewable energy technologies have become increasingly important in modern society. Solar panels and wind turbines can reduce reliance on fossil fuels when supported by reliable energy storage. Governments and institutions are investing in these technologies to reduce emissions. However, adoption also depends on infrastructure, policy, and local economic conditions. A careful evaluation should consider both environmental benefits and practical limitations.`;

    function showAIDetectStatus(message, kind = 'info') {
        if (!aiDetectStatus) return;
        aiDetectStatus.textContent = message;
        aiDetectStatus.className = `ai-status-message ${kind}`;
    }

    function formatProbability(value) {
        return typeof value === 'number' && Number.isFinite(value)
            ? `${(value * 100).toFixed(1)}%`
            : '—';
    }

    function setText(id, value) {
        const element = document.getElementById(id);
        if (element) element.textContent = value === null || value === undefined ? '—' : String(value);
    }

    if (aiDetectText && aiDetectCharCount) {
        const updateAICharCount = () => {
            aiDetectCharCount.innerHTML = `<i class="fa-solid fa-font"></i> ${aiDetectText.value.length.toLocaleString()} characters`;
        };
        aiDetectText.addEventListener('input', updateAICharCount);
        updateAICharCount();
    }

    if (aiDetectLoadSample && aiDetectText) {
        aiDetectLoadSample.addEventListener('click', () => {
            aiDetectText.value = aiDetectSample;
            aiDetectText.dispatchEvent(new Event('input'));
            if (aiDetectStatus) aiDetectStatus.classList.add('hidden');
            if (aiDetectResults) aiDetectResults.classList.add('hidden');
        });
    }

    if (aiDetectClear && aiDetectText) {
        aiDetectClear.addEventListener('click', () => {
            aiDetectText.value = '';
            aiDetectText.dispatchEvent(new Event('input'));
            if (aiDetectStatus) aiDetectStatus.classList.add('hidden');
            if (aiDetectResults) aiDetectResults.classList.add('hidden');
            aiDetectText.focus();
        });
    }

    if (aiDetectBtn && aiDetectText) {
        aiDetectBtn.addEventListener('click', async () => {
            const text = aiDetectText.value.trim();
            if (!text) {
                showAIDetectStatus('Please paste an essay or load the sample before analysis.', 'error');
                return;
            }

            aiDetectBtn.disabled = true;
            if (aiDetectSpinner) aiDetectSpinner.style.display = 'inline-block';
            if (aiDetectResults) aiDetectResults.classList.add('hidden');
            showAIDetectStatus('Analyzing text. The transformer and language-model features may take a little time on the first request.', 'info');

            try {
                const response = await fetch('/api/ai-detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });

                let result;
                try {
                    result = await response.json();
                } catch (_) {
                    throw new Error(`Server returned HTTP ${response.status} without a readable JSON response.`);
                }

                if (!response.ok) {
                    const detail = result.error || 'AI detection request failed.';
                    if (response.status === 503) {
                        throw new Error(`AI detector is not ready. ${detail}`);
                    }
                    throw new Error(detail);
                }

                setText('aiDetectVerdict', result.verdict || (result.is_ai ? 'Likely AI-generated' : 'Likely human-authored'));
                setText('aiDetectRisk', result.risk_level || 'UNKNOWN');
                setText('aiDetectExplanation', result.explanation || 'No explanation was returned.');
                setText('aiDetectAIProb', formatProbability(result.ai_probability));
                setText('aiDetectHumanProb', formatProbability(result.human_probability));
                setText('aiDetectConfidence', formatProbability(result.confidence));
                const metrics = result.metrics || {};
                setText('aiDetectPerplexity', metrics.perplexity ?? '—');
                setText('aiDetectBurstiness', metrics.burstiness_cv ?? '—');
                setText('aiDetectTTR', metrics.lexical_diversity_ttr ?? '—');
                setText('aiDetectSentenceCount', metrics.sentence_count ?? '—');
                setText('aiDetectAvgSentence', metrics.avg_sentence_len ?? '—');

                if (aiDetectSentenceList) {
                    aiDetectSentenceList.replaceChildren();
                    const sentences = Array.isArray(result.sentence_analysis) ? result.sentence_analysis : [];
                    if (!sentences.length) {
                        const empty = document.createElement('p');
                        empty.className = 'subtitle';
                        empty.textContent = 'No sentence-level analysis was returned.';
                        aiDetectSentenceList.appendChild(empty);
                    } else {
                        sentences.forEach(item => {
                            const card = document.createElement('article');
                            card.className = 'ai-sentence-item';
                            const top = document.createElement('div');
                            top.className = 'ai-sentence-top';
                            const status = document.createElement('span');
                            status.className = 'ai-sentence-status';
                            status.textContent = item.status || 'Unclassified';
                            const details = document.createElement('span');
                            details.textContent = `Perplexity: ${item.perplexity ?? '—'} · Words: ${item.word_count ?? '—'}`;
                            const sentence = document.createElement('p');
                            sentence.textContent = item.sentence || '';
                            top.append(status, details);
                            card.append(top, sentence);
                            aiDetectSentenceList.appendChild(card);
                        });
                    }
                }

                if (aiDetectResults) aiDetectResults.classList.remove('hidden');
                if (result.mode === 'trained') {
                    showAIDetectStatus('Analysis completed. Treat this as a model-estimated signal, not proof of authorship.', 'success');
                } else {
                    showAIDetectStatus('The endpoint returned a result, but the model reports that it is not fully trained. Do not rely on these scores.', 'error');
                }
            } catch (error) {
                showAIDetectStatus(error.message || 'AI detection failed. Check the Flask server logs.', 'error');
            } finally {
                aiDetectBtn.disabled = false;
                if (aiDetectSpinner) aiDetectSpinner.style.display = 'none';
            }
        });
    }
});
