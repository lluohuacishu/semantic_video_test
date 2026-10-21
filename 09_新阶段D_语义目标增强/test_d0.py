"""CPU checks for alignment corruption and split leakage."""
import copy
import io
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from prepare_d0 import (decode_union, indexed_annotations, load_json, resolve_rgb,
                        validate_split, validate_supplied_mask)
from audit_d0_timing import classify_matches


class D0DataChecks(unittest.TestCase):
    def setUp(self):
        self.pixels = np.zeros((7, 11), np.uint8)
        self.pixels[2:5, 4:9] = 1
        rle = coco_mask.encode(np.asfortranarray(self.pixels))
        rle['counts'] = rle['counts'].decode('ascii')
        self.frame = {'id': 1, 'width': 11, 'height': 7, 'file_name': '000002.png'}
        self.instance = {'id': 3, 'image_id': 1, 'category_id': 1, 'iscrowd': 0,
                         'segmentation': rle, 'area': 15, 'bbox': [4, 2, 4, 2]}

    def test_original_filename_not_ordinal(self):
        name = 'SFISHTRACK/Frames/video_027/000002.png'
        self.assertEqual(resolve_rgb({name}, 27, '000002.png'), name)

    def test_known_jpeg_alias(self):
        name = 'SFISHTRACK/Frames/video_001/frame_000002.jpg'
        self.assertEqual(resolve_rgb({name}, 1, '000002.png'), name)

    def test_ambiguous_alias_rejected(self):
        names = {'SFISHTRACK/Frames/video_001/000002.png',
                 'SFISHTRACK/Frames/video_001/frame_000002.jpg'}
        with self.assertRaises(ValueError):
            resolve_rgb(names, 1, '000002.png')

    def test_missing_or_nonlocal_filename_rejected(self):
        for filename in ('000002.png', '../000002.png'):
            with self.assertRaises(ValueError):
                resolve_rgb(set(), 1, filename)

    def test_nonsquare_compressed_rle_keeps_orientation(self):
        decoded = decode_union(self.frame, [self.instance])
        np.testing.assert_array_equal(decoded, self.pixels.astype(bool))

    def test_wrong_area_bbox_or_dimensions_rejected(self):
        for field, value in [('area', 14), ('bbox', [2, 4, 5, 3])]:
            bad = copy.deepcopy(self.instance)
            bad[field] = value
            with self.assertRaises(ValueError):
                decode_union(self.frame, [bad])
        with self.assertRaises(ValueError):
            decode_union(dict(self.frame, width=10), [self.instance])

    def test_empty_target_is_valid(self):
        self.assertFalse(decode_union(self.frame, []).any())

    def test_supplied_mask_detects_single_pixel_corruption(self):
        supplied = np.repeat((self.pixels * 128)[:, :, None], 3, axis=2)
        buffer = io.BytesIO()
        Image.fromarray(supplied).save(buffer, format='PNG')
        validate_supplied_mask(buffer.getvalue(), self.pixels.astype(bool))
        supplied[0, 0] = 128
        buffer = io.BytesIO()
        Image.fromarray(supplied).save(buffer, format='PNG')
        with self.assertRaises(ValueError):
            validate_supplied_mask(buffer.getvalue(), self.pixels.astype(bool))

    def test_unbound_or_duplicate_annotations_rejected(self):
        data = {'images': [self.frame], 'annotations': [self.instance],
                'categories': [{'id': 1, 'name': 'object'}]}
        self.assertEqual(len(indexed_annotations(data)[1]), 1)
        for invalid in [dict(data, images=[self.frame, self.frame]),
                        dict(data, annotations=[dict(self.instance, image_id=2)])]:
            with self.assertRaises(ValueError):
                indexed_annotations(invalid)

    def test_whole_sequence_split_and_history_rules(self):
        config = load_json(Path(__file__).with_name('d0_selection.json'))
        validate_split(config)
        invalid = copy.deepcopy(config)
        invalid['reserved_group_video_ids'].append(1)
        with self.assertRaises(ValueError):
            validate_split(invalid)
        invalid = copy.deepcopy(config)
        invalid['fish_reserved_video_ids'] = [27]
        with self.assertRaises(ValueError):
            validate_split(invalid)

    def test_dense_constant_timing_ready(self):
        matches = [{'accepted': True, 'best_source_frame_index': i} for i in range(3)]
        timestamps = [{'best_effort_timestamp': i * 1001} for i in range(3)]
        result = classify_matches(matches, timestamps, Fraction(1, 30000))
        self.assertTrue(result['ready_for_dense_video_comparison'])
        self.assertEqual(result['fps_if_constant'], '30000/1001')

    def test_vfr_is_not_silently_marked_constant(self):
        matches = [{'accepted': True, 'best_source_frame_index': i} for i in range(3)]
        timestamps = [{'best_effort_timestamp': i} for i in [0, 19, 40]]
        result = classify_matches(matches, timestamps, Fraction(1, 300))
        self.assertTrue(result['dense_source_frames'])
        self.assertFalse(result['ready_for_dense_video_comparison'])
        self.assertIsNone(result['fps_if_constant'])

    def test_ambiguous_or_skipped_matches_not_ready(self):
        timestamps = [{'best_effort_timestamp': i * 1001} for i in range(4)]
        for matches in ([{'accepted': True, 'best_source_frame_index': i} for i in [0, 2, 3]],
                        [{'accepted': i != 1, 'best_source_frame_index': i} for i in range(3)]):
            result = classify_matches(matches, timestamps, Fraction(1, 30000))
            self.assertFalse(result['ready_for_dense_video_comparison'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
