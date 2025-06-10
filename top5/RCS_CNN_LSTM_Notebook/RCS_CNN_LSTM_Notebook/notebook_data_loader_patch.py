# --- Correct alignment: align on original index, drop NaNs after alignment, do not reset_index before alignment ---

# features_sel and target should both have DatetimeIndex
# Do NOT reset_index or dropna before alignment
common_index = features_sel.index.intersection(target.index)
features_sel_aligned = features_sel.loc[common_index]
target_aligned = target.loc[common_index]

# Drop any rows with NaN in features or target after alignment
aligned = features_sel_aligned.join(target_aligned.rename("target")).dropna()
features_final = aligned.drop(columns=["target"])
target_final = aligned["target"]

print("features_final shape:", features_final.shape)
print("target_final shape:", target_final.shape)
