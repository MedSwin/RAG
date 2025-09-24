
import pandas as pd

def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Simply remove all the data row that dont have the input and output

    Args:
        df (pd.DataFrame): data frame needs to be clean

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    try:
        if not df.empty:
            df.dropna(subset=["input", "output"], inplace=True)
            print(f"After dropping NA: {len(df)} rows")

            truncate_pattern = r'^Hi[,.]?\s*\.{0,3}\s*$|^Hello[,.]?\s*\.{0,3}\s*$|^Welcome[,.]?\s*\.{0,3}\s*$'
            df = df[~df['input'].str.match(truncate_pattern, na=False)]
            print(f"After removing truncated outputs because they are only 'welcome' words: {len(df)} rows")
            
            df = df[df['input'].str.strip().str.len() > 10]
            df = df[df['output'].str.strip().str.len() > 10]
            
            # df["input_length"] = df["input"].str.len()
            # df["output_length"] = df["output"].str.len()
            
            print(f"After removing short inputs/outputs: {len(df)} rows")

            print(f"Average input length: {df['input_length'].mean()} characters")
            print(f"Average output length: {df['output_length'].mean()} characters")
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"If this problem {e} is raised, check the data preprocessing")
    return df





