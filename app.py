"""Presentation only; services own ingestion, analytics and generation."""
import json
from pathlib import Path
import streamlit as st
from cctv_rag.runtime import Runtime
from cctv_rag.query_parser import Clarification
from cctv_rag.analytics import vehicle_analytics_overview


def main():
    st.set_page_config(page_title='CCTV Record Analytics', layout='wide')
    st.title('CCTV Record Analytics')
    st.caption('Development component. Connects to SQLite/PostgreSQL storage, deterministic analytics, and optional Gemini/MiniLM.')
    try:
        runtime = Runtime()
    except Exception:
        st.error('Startup failed. Check STORAGE_BACKEND, DATABASE_URL, TIME_ZONE, optional provider configuration, and installed dependencies. See README. Credentials are not displayed.')
        return
    try:
        # System & AI Service Status
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric('Storage Backend', runtime.config.backend.upper())
        with stat_col2:
            if runtime.embedder is not None:
                st.metric('Embeddings', 'MiniLM (Active)')
            elif runtime.config.embedding_mode == 'minilm':
                st.metric('Embeddings', 'MiniLM (Unavailable/Offline)')
            else:
                st.metric('Embeddings', 'Disabled (Metadata Only)')
        with stat_col3:
            if runtime.llm is not None:
                st.metric('Gemini LLM', f'Configured ({runtime.config.gemini_model})')
            else:
                st.metric('Gemini LLM', 'Disabled (Deterministic Only)')

        if getattr(runtime, 'embedder_error', None):
            st.warning(runtime.embedder_error)

        st.info(f'Storage: {runtime.config.backend}. Embeddings: {runtime.config.embedding_mode}. '
                'Without embeddings, retrieval uses camera/time metadata only. Without Gemini, answers use deterministic evidence.')
        st.warning('Coverage is unknown. Recorded counts are not complete traffic totals. Synthetic data is test data.')

        with st.expander('Development ingestion', expanded=False):
            batch_choice = st.selectbox('Synthetic fixture', ['batch_001', 'batch_002'])
            if st.button('Ingest selected synthetic batch'):
                data = json.loads((Path(__file__).parent / 'sample_data' / (batch_choice+'.json')).read_text())
                st.json(runtime.ingestor.ingest_batch(data, batch_choice))
            uploaded = st.file_uploader('Validated JSON batch', type=['json'])
            batch_id = st.text_input('Stable batch ID')
            if st.button('Ingest uploaded batch'):
                if uploaded is None or not batch_id.strip():
                    st.error('Choose a JSON file and enter its stable batch ID.')
                else:
                    try:
                        st.json(runtime.ingestor.ingest_batch(json.load(uploaded), batch_id))
                    except (ValueError, TypeError):
                        st.error('Invalid JSON input or batch ID.')
        st.subheader('Ingestion status')
        st.json(runtime.status())

        st.subheader('Vehicle Traffic & Directional Analytics')
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            cam_filter = st.text_input('Camera filter (e.g. CAM_01)', '', key='analytics_cam')
        with col_ctrl2:
            src_filter = st.selectbox('Data source', ['all', 'live', 'synthetic'], key='analytics_src')
        with col_ctrl3:
            time_filter = st.checkbox('Filter by UTC time window', value=False, key='analytics_time_enable')

        start_utc, end_utc = None, None
        if time_filter:
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                start_utc = st.text_input('Window start (UTC ISO)', '2026-10-09T00:00:00Z', key='analytics_start')
            with t_col2:
                end_utc = st.text_input('Window end (UTC ISO)', '2026-10-12T00:00:00Z', key='analytics_end')

        overview = vehicle_analytics_overview(
            runtime.store,
            start=start_utc if time_filter else None,
            end=end_utc if time_filter else None,
            camera=cam_filter.strip() or None,
            source=None if src_filter == 'all' else src_filter
        )

        if overview['valid_records'] == 0:
            st.info('No events match the selected filters. Ingest CCTV batches or adjust the camera, source, or time filters.')
        else:
            if overview['is_mixed_source']:
                st.warning('⚠️ Showing mixed data: contains both live and synthetic events. Synthetic data is test data.')
            elif 'live' in overview['source_types']:
                st.success('🟢 Source: Live CCTV observations only.')
            elif 'synthetic' in overview['source_types']:
                st.info('🧪 Source: Synthetic test data only.')

            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric('Total Crossings', overview['total_vehicle_crossings'])
            m_col2.metric('Cars', overview['by_subtype'].get('car', 0))
            m_col3.metric('Trucks', overview['by_subtype'].get('truck', 0))
            m_col4.metric('Buses', overview['by_subtype'].get('bus', 0))
            m_col5.metric('Motorcycles', overview['by_subtype'].get('motorcycle', 0))

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown('**Crossings by Vehicle Subtype**')
                st.table([
                    {'Subtype': k.capitalize(), 'Crossings': v}
                    for k, v in sorted(overview['by_subtype'].items())
                    if v > 0 or k in ['car', 'truck', 'bus', 'motorcycle']
                ])
            with b_col2:
                st.markdown('**Crossings by Direction**')
                st.table([
                    {'Direction': k.capitalize(), 'Crossings': v}
                    for k, v in sorted(overview['by_direction'].items())
                    if v > 0 or k in ['positive', 'negative']
                ])

            st.markdown('**Subtype × Direction Cross-tabulation**')
            crosstab_rows = []
            has_unspec_dir = any(overview['crosstab'].get(st_name, {}).get('unspecified', 0) > 0 for st_name in overview['crosstab'])
            for st_name in ['car', 'truck', 'bus', 'motorcycle', 'unspecified']:
                pos_c = overview['crosstab'].get(st_name, {}).get('positive', 0)
                neg_c = overview['crosstab'].get(st_name, {}).get('negative', 0)
                unspec_c = overview['crosstab'].get(st_name, {}).get('unspecified', 0)
                sub_total = pos_c + neg_c + unspec_c
                if sub_total > 0 or st_name in ['car', 'truck', 'bus', 'motorcycle']:
                    row = {
                        'Vehicle Subtype': st_name.capitalize(),
                        'Positive': pos_c,
                        'Negative': neg_c,
                    }
                    if has_unspec_dir:
                        row['Unspecified Dir'] = unspec_c
                    row['Total'] = sub_total
                    crosstab_rows.append(row)
            grand_row = {
                'Vehicle Subtype': 'Grand Total',
                'Positive': overview['by_direction'].get('positive', 0),
                'Negative': overview['by_direction'].get('negative', 0),
            }
            if has_unspec_dir:
                grand_row['Unspecified Dir'] = overview['by_direction'].get('unspecified', 0)
            grand_row['Total'] = overview['total_vehicle_crossings']
            crosstab_rows.append(grand_row)
            st.table(crosstab_rows)

        st.subheader('Natural Language Query')
        question = st.text_input('Question', 'How many vehicle crossings did CAM_01 record on October 9, 2026?')
        camera = st.text_input('Optional camera filter', '')
        source = st.selectbox('Source', ['all', 'synthetic', 'live'])
        explicit = st.checkbox('Use explicit UTC start/end filters')
        start = st.text_input('Start (inclusive ISO-8601 UTC)', '2026-10-09T00:00:00Z') if explicit else None
        end = st.text_input('End (exclusive ISO-8601 UTC)', '2026-10-11T00:00:00Z') if explicit else None
        if st.button('Ask'):
            try:
                result = runtime.service.ask(question, camera=camera.strip() or None, start=start, end=end,
                                             source=None if source == 'all' else source)
                st.subheader('Answer')
                if result.get('generation') == 'gemini':
                    st.success('🤖 Grounded Gemini Explanation')
                else:
                    st.info('📊 Deterministic Analytics / Structured Evidence')

                st.write(result['answer'])
                st.caption(result['interpretation'])
                for key in ('service_warning', 'retrieval_warning', 'window_notice'):
                    if key in result:
                        st.warning(result[key])
                if result['analytics'] is not None:
                    st.json(result['analytics'])
                with st.expander('Current summary evidence'):
                    evidence_list = result.get('evidence', [])
                    if evidence_list and any('score' in r for r in evidence_list):
                        st.caption('🔍 Semantic Retrieval Results (ranked by cosine similarity):')
                        st.table([
                            {
                                'Summary ID': (r.get('id') or '')[:12] + '...',
                                'Camera': r.get('camera_id'),
                                'Window Start': r.get('start'),
                                'Relevance Score': f"{r.get('score', 0):.4f}",
                                'Summary': r.get('text'),
                            }
                            for r in evidence_list
                        ])
                    else:
                        st.json(evidence_list)
            except (Clarification, ValueError) as exc:
                st.warning(str(exc))
            except Exception:
                st.error('The request could not be completed. Check service availability; no successful answer was generated.')
    except Exception:
        st.error('Storage or ingestion is unavailable. Retry after checking the server configuration; the operation was not confirmed successful.')
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
