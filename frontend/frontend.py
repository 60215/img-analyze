import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import numpy as np
import altair as alt

FASTAPI_URL = "http://localhost:8000"

if "page" in st.query_params:
    st.session_state.page = st.query_params["page"]
elif "page" not in st.session_state:
    st.session_state.page = "Upload"

if "id" in st.query_params:
    st.session_state.current_id = st.query_params["id"]
elif "current_id" not in st.session_state:
    st.session_state.current_id = ""

st.sidebar.title("Navigation")
pages = ["Upload", "Analyze"]

current_page = st.session_state.page if st.session_state.page in pages else "Upload"
selected_page = st.sidebar.radio("Go to", pages, index=pages.index(current_page))

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.query_params["page"] = selected_page

    if selected_page == "Upload":
        st.session_state.current_id = ""
        st.query_params.pop("id", None)

    st.rerun()

if st.session_state.page == "Upload":
    st.title("Upload image to be analyzed")
    st.write("Upload your image to get an analysis.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Image Preview', use_container_width=True)

        if st.button("Process Image"):
            with st.spinner("Analyzing image..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{FASTAPI_URL}/upload/", files=files)

                    if response.status_code == 200:
                        data = response.json()
                        st.success("Success!")

                        # --- 3. Update state and URL on successful upload ---
                        st.session_state.current_id = data["id"]
                        st.session_state.page = "Analyze"
                        st.query_params["id"] = data["id"]
                        st.query_params["page"] = "Analyze"

                        st.rerun()
                    else:
                        st.error(f"Error: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection Error: Is the FastAPI backend running?")

elif st.session_state.page == "Analyze":

    # Specific id selected
    if st.session_state.current_id:
        if st.button("Back to Gallery"):
            st.session_state.current_id = ""
            st.query_params.pop("id", None)
            st.rerun()

        image_id = st.session_state.current_id
        st.title(f"Analysis for Image")
        st.caption(f"ID: {image_id}")

        meta_res = requests.get(f"{FASTAPI_URL}/analyze/{image_id}")

        st.divider()
        st.subheader("Views analysis")

        # TODO: тянуть с метадаты тут все, когда будет модель.
        np.random.seed(hash(image_id) % (2 ** 32 - 1))
        historical_views = [np.random.randint(200, 500)]
        for _ in range(6):
            historical_views.append(historical_views[-1] + np.random.randint(50, 250))

        last_actual_view = historical_views[-1]
        predicted_views = last_actual_view + np.random.randint(200, 500)

        days_hist = [f"Day {-6 + i}" for i in range(7)]
        day_pred = "Tomorrow"

        df_hist = pd.DataFrame({
            'Day': days_hist,
            'Views': historical_views,
            'Type': 'History'
        })

        df_pred = pd.DataFrame({
            'Day': [days_hist[-1], day_pred],
            'Views': [last_actual_view, predicted_views],
            'Type': 'Prediction'
        })

        df_chart = pd.concat([df_hist, df_pred])
        sort_order = days_hist + [day_pred]

        chart = alt.Chart(df_chart).mark_line(point=True).encode(
            x=alt.X('Day', sort=sort_order, title=""),
            y=alt.Y('Views', title="Views"),
            color=alt.Color('Type', scale=alt.Scale(domain=['History', 'Prediction'], range=['#3498DB', '#E74C3C'])),
            strokeDash=alt.condition(
                alt.datum.Type == 'Prediction',
                alt.value([5, 5]),
                alt.value([0])
            )
        ).properties(
            height=350
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        if meta_res.status_code == 200:
            metadata = meta_res.json()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Object Tags")
                for tag in metadata['object_tags']:
                    st.progress(tag['probability'], text=f"{tag['tag']} ({tag['probability'] * 100:.1f}%)")

            with col2:
                st.subheader("Vibe Tags")
                for tag in metadata['vibe_tags']:
                    st.progress(tag['probability'], text=f"{tag['tag']} ({tag['probability'] * 100:.1f}%)")

            img_res = requests.get(f"{FASTAPI_URL}/image/{image_id}")
            if img_res.status_code == 200:
                image = Image.open(io.BytesIO(img_res.content))
                st.image(image, caption=metadata.get("filename", "Uploaded Image"), use_container_width=True)
        else:
            st.error("Analysis data not found for this ID.")

    # No tag - gallery view
    else:
        st.title("Gallery")
        st.write("Click on image to get analysis")

        response = requests.get(f"{FASTAPI_URL}/images/")

        images_data = response.json()

        if not images_data:
            st.info("No images in the storage currently.")
        else:
            cols = st.columns(3)

            for index, img_info in enumerate(images_data):
                col_idx = index % 3
                with cols[col_idx]:
                    img_id = img_info["id"]
                    thumb_res = requests.get(f"{FASTAPI_URL}/image/{img_id}")

                    thumb_img = Image.open(io.BytesIO(thumb_res.content))
                    st.image(thumb_img, use_container_width=True)

                    short_title = img_info['filename'] if len(
                        img_info['filename']) <= 20 else f"{img_info['filename'][:17]}..."
                    st.write(f"**{short_title}**")

                    if st.button("View Details", key=f"btn_{img_id}"):
                        st.session_state.current_id = img_id
                        st.query_params["id"] = img_id
                        st.rerun()