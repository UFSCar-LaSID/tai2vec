
import scripts as kw
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def preprocess_amazon_games(input_path: str, output_path: str):
    '''
    Preprocesses the Amazon Games dataset.

    params:
        input_path: Path to the input directory containing the dataset files.
        output_path: Path to the output directory where the processed files will be saved.
    '''


    os.makedirs(output_path, exist_ok=True)

    import json

    with open(os.path.join(input_path, 'Video_Games.jsonl'), 'r') as fp:
        reviews = fp.readlines()
    for i, line in tqdm(enumerate(reviews), total=len(reviews)):
        reviews[i] = json.loads(reviews[i].strip())
    df_interactions = pd.DataFrame.from_records(reviews)

    df_interactions = df_interactions.reset_index().rename(
        columns={'index': 'id_review', 'rating': kw.COLUMN_RATING, 'parent_asin': kw.COLUMN_ITEM_ID,
                'asin': 'id_item_style', 'user_id': kw.COLUMN_USER_ID, 'timestamp': kw.COLUMN_TIMESTAMP}
    )

    df_interactions['images_count'] = df_interactions['images'].progress_apply(len)
    df_images_reviews = list()
    reviews_with_image = df_interactions[df_interactions['images_count']>0].copy()
    for _, row in tqdm(reviews_with_image.iterrows(), total=len(reviews_with_image)):
        for image_url in row['images']:
            image_url['id_review'] = row['id_review']
            df_images_reviews.append(image_url)
    del reviews_with_image
    df_images_reviews = pd.DataFrame.from_records(df_images_reviews).drop(columns=['attachment_type'])
    df_images_reviews = df_images_reviews[['id_review']+list(df_images_reviews.columns)[:-1]]
    df_interactions.drop(columns=['images_count', 'images'], inplace=True)

    df_interactions[kw.COLUMN_RATING] = df_interactions[kw.COLUMN_RATING].astype(int)
    df_interactions[kw.COLUMN_ITEM_ID] = df_interactions[kw.COLUMN_ITEM_ID].astype(str)
    df_interactions['id_item_style'] = df_interactions['id_item_style'].astype(str)
    df_interactions[kw.COLUMN_USER_ID] = df_interactions[kw.COLUMN_USER_ID].astype(str)
    df_interactions['verified_purchase'] = df_interactions['verified_purchase'].astype(int)

    from datetime import datetime
    df_interactions[kw.COLUMN_TIMESTAMP] = df_interactions[kw.COLUMN_TIMESTAMP]//1000
    df_interactions[kw.COLUMN_DATETIME] = df_interactions[kw.COLUMN_TIMESTAMP].progress_apply(datetime.fromtimestamp)

    main_cols = [kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID, kw.COLUMN_RATING, kw.COLUMN_DATETIME, kw.COLUMN_TIMESTAMP]
    secondary_cols = [col for col in df_interactions.columns if col not in main_cols]
    df_interactions = df_interactions[main_cols+secondary_cols]

    for df, file_name in tqdm([(df_interactions, kw.FILE_INTERACTIONS)]):
        df.to_csv(os.path.join(output_path, file_name), sep=kw.DELIMITER, encoding=kw.ENCODING, quoting=kw.QUOTING, quotechar=kw.QUOTECHAR, header=True, index=False)