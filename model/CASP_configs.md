## Learning rate tests

```json
{
    "experiments": [
        {
            "base_name": "se_casp_lr",
            "dataset_name": "casp",
            "lr": [
                1e-2,
                5e-3,
                1e-3,
                5e-4,
                1e-4,
                5e-5,
                1e-5,
                5e-6,
                1e-6
            ],
            "loss_type": "gaussian_se",
            "hidden_dims": [
                64,
                64,
                64
            ]
        },
        {
            "base_name": "kernel_casp_lr",
            "dataset_name": "casp",
            "lr": [
                1e-2,
                5e-3,
                1e-3,
                5e-4,
                1e-4,
                5e-5,
                1e-5,
                5e-6,
                1e-6
            ],
            "loss_type": "gaussian_kernel",
            "hidden_dims": [
                64,
                64,
                64
            ]
        },
        {
            "base_name": "nll_casp_lr",
            "dataset_name": "casp",
            "lr": [
                1e-2,
                5e-3,
                1e-3,
                5e-4,
                1e-4,
                5e-5,
                1e-5,
                5e-6,
                1e-6
            ],
            "loss_type": "gaussian_nll",
            "hidden_dims": [
                64,
                64,
                64
            ]
        },
        {
            "base_name": "crps_casp_lr",
            "dataset_name": "casp",
            "lr": [
                1e-2,
                5e-3,
                1e-3,
                5e-4,
                1e-4,
                5e-5,
                1e-5,
                5e-6,
                1e-6
            ],
            "loss_type": "gaussian_crps",
            "hidden_dims": [
                64,
                64,
                64
            ]
        }
    ],
    "do_not_vary": [
        "hidden_dims"
    ]
}
```

## dropout-gamma tests

```json
{
    "experiments": [
        {
            "base_name": "se_casp_doga",
            "dataset_name": "casp",
            "lr": 1e-3,
            "dropout": [0.05, 0.1, 0.2, 0.3],
            "loss_type": "gaussian_se",
            "hidden_dims": [
                64,
                64,
                64
            ]
        },
        {
            "base_name": "kernel_casp_doga",
            "dataset_name": "casp",
            "lr": 1e-3,
            "dropout": [0.05, 0.1, 0.2, 0.3],
            "loss_type": "gaussian_kernel",
            "loss_gk_gamma": [0.5, 1, 2, 4],
            "hidden_dims": [
                64,
                64,
                64
            ]
        },
        {
            "base_name": "nll_casp_doga",
            "dataset_name": "casp",
            "lr": 5e-3,
            "dropout": [0.05, 0.1, 0.2, 0.3],
            "loss_type": "gaussian_nll",
            "hidden_dims": [
                64,
                64,
                64
            ]
        },
        {
            "base_name": "crps_casp_doga",
            "dataset_name": "casp",
            "lr": 1e-3,
            "dropout": [0.05, 0.1, 0.2, 0.3],
            "loss_type": "gaussian_crps",
            "hidden_dims": [
                64,
                64,
                64
            ]
        }
    ],
    "do_not_vary": [
        "hidden_dims"
    ]
}
```
