"""Presentation only; services own ingestion, analytics and generation."""
import json
from pathlib import Path
import streamlit as st
from cctv_rag.runtime import Runtime
from cctv_rag.query_parser import Clarification


def main():
    st.set_page_config(page_title='CCTV Record Analytics', layout='wide')
    st.title('CCTV Record Analytics')
    st.caption('Development component. No cameras or AWS pipeline are connected by this application.')
    try:
        runtime = Runtime()
    except Exception:
        st.error('Startup failed. Check STORAGE_BACKEND, DATABASE_URL, TIME_ZONE, optional provider configuration, and installed dependencies. See README. Credentials are not displayed.')
        return
    try:
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
                st.write(result['answer'])
                st.caption(result['interpretation'])
                for key in ('service_warning', 'retrieval_warning', 'window_notice'):
                    if key in result:
                        st.warning(result[key])
                if result['analytics'] is not None:
                    st.json(result['analytics'])
                with st.expander('Current summary evidence'):
                    st.json(result['evidence'])
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
