from pathlib import Path
import importlib.util
import unittest
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CorrectedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.m = load(ROOT/'code/generate_corrected.py', 'corrected_generator')
        self.ss = load(ROOT/'single_slot.py', 'existing_single_slot')
        self.original = Image.fromarray(np.random.default_rng(91).integers(0, 127, (100, 180, 3), dtype=np.uint8))
        self.crop = Image.fromarray(np.random.default_rng(93).integers(0, 255, (30, 20, 3), dtype=np.uint8))
        self.target = {'target_id':'TEST', 'original_size':[180,100], 'canvas_size':[228,100],
                       'target_bbox_xyxy_exclusive':[65,20,115,80], 'R_xyxy_exclusive':[72,29,108,72],
                       'P_xyxy_exclusive':[119,20,155,63], 'S_xyxy_exclusive':[190,20,226,63]}

    def test_actual_original_renderer_parity_and_minimal_edits(self):
        for ratio in (.6,.8,1.):
            r=self.ss.geometry.scaled_r(self.target['R_xyxy_exclusive'],ratio)
            for p in self.ss.geometry.position_slots(self.target,r).values():
                for condition in ('O','G','T','N-near'):
                    im=self.m.render(self.ss.renderer,self.original,self.target,self.crop,r,p,condition)
                    expected=self.ss.render(self.original,self.target,self.crop,r,p,condition)
                    self.assertEqual(im.tobytes(),expected.tobytes())
                    self.m.verify_minimal(self.ss.renderer,im,self.original,self.target,r,p,condition)
                    damaged=im.copy();damaged.putpixel((0,99),(128,128,128))
                    with self.assertRaises(AssertionError):
                        self.m.verify_minimal(self.ss.renderer,damaged,self.original,self.target,r,p,condition)

    def test_requests_and_pairs_are_position_matched(self):
        requests,pairs=self.m.design_ids('TEST',['D1','D2'])
        self.assertEqual(len(requests),78)
        self.assertEqual(len(set(requests)),78)
        self.assertEqual(len(pairs),24)
        for pair in pairs:
            for label in ('O','G','T','N'):
                self.assertTrue(pair[label+'_request_id'].endswith('__'+pair['position']))
            self.assertNotIn(pair['position'],pair['donor_check_request_id'])
        self.assertEqual(len({p['T_request_id'] for p in pairs}),24)
        self.assertEqual(len({p['O_request_id'] for p in pairs}),12)
        self.assertIn('v2__TEST__s060__D1__N__left_top',requests)
        self.assertEqual(self.m.request_id('TEST','s060','N-near','left_top','D1'),'v2__TEST__s060__D1__N__left_top')

    def test_human_pixel_mismatch_rejected(self):
        im=Image.new('RGB',(4,5),'red')
        ref={'rgb_sha256':self.m.pixel_hash(im), 'rgba_sha256':self.m.pixel_hash(im,'RGBA'), 'image_size':[4,5]}
        self.m.verify_human_pixels(im,ref)
        im.putpixel((0,0),(0,0,0))
        with self.assertRaises(AssertionError):self.m.verify_human_pixels(im,ref)

    def test_full_design_and_all_actual_human_ids(self):
        targets=self.m.read(ROOT/'source_support/selected_targets.jsonl')
        requests=[];pairs=[]
        for target in targets:
            ids,ps=self.m.design_ids(target['target_id'],target['donor_ids'])
            for rid in ids:
                bits=rid.split('__');has_donor=bits[3].startswith('D')
                condition=bits[4] if has_donor else bits[3]
                condition='N-near' if condition=='N' else condition
                pos=None if condition=='donor-check' else bits[-1]
                requests.append({'request_id':rid,'target_id':target['target_id'],'level':bits[2],
                    'condition':condition,'position':pos,'donor_id':bits[3] if has_donor else None,
                    'R_xyxy_exclusive':[1,1,3,3],'P_xyxy_exclusive':[4,1,6,3] if pos else None})
            for pair in ps:pair.update(R_xyxy_exclusive=[1,1,3,3],P_xyxy_exclusive=[4,1,6,3])
            pairs.extend(ps)
        lookup=self.m.verify_design(requests,pairs,targets)
        selected=self.m.read(ROOT/'human_annotation_v2/coordinator_private/selected_stimuli.jsonl')
        self.assertEqual(len(selected),72)
        for row in selected:
            self.assertIn(row['request_id'],lookup)
            self.assertEqual(lookup[row['request_id']]['condition'],'N-near' if row['condition']=='N' else row['condition'])
        damaged=[dict(row) for row in pairs]
        damaged[0]['T_request_id']=damaged[2]['T_request_id']
        with self.assertRaises(AssertionError):self.m.verify_design(requests,damaged,targets)


if __name__=='__main__':unittest.main()
