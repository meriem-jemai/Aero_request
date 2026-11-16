from gym.envs.registration import register

# Environment with pixel observations
register(
    id="Aerorequest_pixels_v0",
    entry_point="Aerorequest.env.aerorequest:Aerorequest",
    kwargs={"obs_type": "pixels"},
)

# Environment with feature observations
register(
    id="Aerorequest_features_v0",
    entry_point="Aerorequest.env.aerorequest:Aerorequest",
    kwargs={"obs_type": "features"},
)