
Code for the paper [Efficient Action Localization with Approximately Normalized Fisher Vectors](http://hal.inria.fr/hal-00979594/PDF/efficient_action_localization.pdf).

## Dependencies

* The [`fisher_vectors`](https://github.com/danoneata/fisher_vectors) module. The data stored at `stats_path` (see the function `load_sample_data` from `load_data.py`) are not exactly Fisher vectors, but an intermediate representation, *sufficient statistics*. To get the sufficient statistics from low-level descriptors, use the `descs_to_sstats` function from [`fv_model.py`](https://github.com/danoneata/fisher_vectors/blob/master/model/fv_model.py); to get the Fisher vectors from sufficient statistics use `sstats_to_features` from [`fv_model.py`](https://github.com/danoneata/fisher_vectors/blob/master/model/fv_model.py). TODO Instead of using the entire module, just isolate those functions.

* You need to define a `Dataset` factory that produces an object instance `dataset`. The `dataset` behaves like a container of the data information (like paths and other similar information). Among others the `dataset` should know how to responds to following attributes and methods:
 - `D` int attribute representing the dimension of the low-level descriptor,
 - `VOC_SIZE` int attribute representing the number of GMM components (the vocabulary size),
 - `GMM` str attribute indicating the path to the GMM object.
 - `SSTATS_DIR` str attribute indicating the path to the sufficient statistics.
 - `get_data` a method that takes a string representing the data split (it can be either `train` or `test`) and returns the video names and their corresponding labels.
TODO Add example of Dataset.

## Code to reproduce the results

### Action recognition experiments

Experiments for the exact normalizations:

    for s in none exact; do
        for l in none exact; do
            python cvpr14camera_ready.py -d hollywood2.delta_5 --e_std_1 --sqrt $s --l2_norm $l -vv
        done
    done

Experiments for the approximate square rooting:

    python cvpr14camera_ready.py -d hollywood2.delta_5 --e_std_1 --sqrt approx --e_std_2 --l2_norm exact -vv

Experiments for approximating the both square rooting and the L2 normalization:

for i in 1 2 4 8 16 32; do
       python cvpr14camera_ready.py -d hollywood2.delta_5 --e_std_1 --sqrt approx --e_std_2 --l2_norm approx -n $i -vv
done

The `evaluate.py` script is a simpler version that does evaluation for action recognition, but it doesn't support more complicated data, _i.e_, spatial pyramids and spatial Fisher vectors.

### Temporal action localization experiments

Experiments for the temporal action localization case.

    for d in cc.no_stab duch.no_stab; do 
      for c in 1 2; do
        for a in "e_std_1.fast" "exact_L2+e_std_1" "exact_sqrt+e_std_1" "exact+e_std_1" "approx+e_std_1"; do
          python -u ~/experiments/normalizations_approximation/detection.py \
              -d $d \
              -a $a \
              --stride 5 \
              --begin 20 \
              -D 5 \
              --end 180 \
              --class_idx $c \
              --rescore \
              -w
        done
      done
    done

