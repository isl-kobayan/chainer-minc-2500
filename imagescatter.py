#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import itertools
import scipy.spatial.distance as dis
from sompy import SOM
import numpy as np
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
import pandas as pd
import dominate
from dominate.tags import *
import math

def colorfloat2hex(color):
    return "'#{0:02x}{1:02x}{2:02x}'".format(
        int(color[0]*255), int(color[1]*255), int(color[2]*255))

def main(args):
    lists = pd.read_csv(args.infile, header=None, delimiter='\t')
    categories = utils.io.load_categories(args.categories)
    C = len(categories)
    x_min, x_max = lists[2].min(), lists[2].max()
    y_min, y_max = lists[3].min(), lists[3].max()

    doc = dominate.document(title='t-SNE browser ' + args.infile)

    with doc.head:
        #link(rel='stylesheet', href='https://cdnjs.cloudflare.com/ajax/libs/c3/0.4.11/c3.min.css')
        link(rel='stylesheet', href='c3.css')
        script(type='text/javascript', src='https://cdn.jsdelivr.net/d3js/latest/d3.min.js')
        script(type='text/javascript', src='https://cdnjs.cloudflare.com/ajax/libs/c3/0.4.11/c3.min.js')
    with doc:
        '''with div(id='categorylist').add(ul(style="list-style:none;")):
            for i, c in enumerate(categories):
                _li = li()
                _label = _li.add(label())
                _label += dominate.tags.input(
                    type="checkbox", id='check_'+c, cls='category_checklist',
                    name="categories[]", value=str(i), checked="checked")
                _label += c'''
        with div(id="chart"):
            _script = script(type='text/javascript')
            xs_str = ",\n".join([c+": '"+c+"_x'" for c in categories])
            color_list = ", ".join([colorfloat2hex(plt.cm.jet(float(i)/(C-1))) for i in range(C)])
            #print(color_list)
            #print(xs_str)
            columns_str_list=[]
            image_path_list=[]
            for i, c in enumerate(categories):
                c_list = lists[lists[1] == i]
                columns_str_list.append(
                "['"+c+"_x', " + ",".join([str(v) for v in c_list[2]]) + "]"
                )
                columns_str_list.append(
                "['"+c+"', " + ",".join([str(v) for v in c_list[3]]) + "]"
                )
                image_path_list.append(
                "imagepath['"+c+"'] = [" + ",".join(["'" + os.path.join(args.root, v) + "'" for v in c_list[0]]) + "];"
                )

            columns_str = ",\n".join(columns_str_list)
            #print(columns_str_list)
            #print(image_path_list)
            append_imagepath_code = "\n".join(image_path_list)
            _script.add_raw_string('var abc = 0;')
            _script.add_raw_string( # += \
'''
var imagepath = {};\n''' + append_imagepath_code + '''
console.log(imagepath['brick']);
function printProperties(obj) {
    var properties = '';
    for (var prop in obj){
        properties += prop + '=' + obj[prop] + '\\n';
    }
    console.log(properties);
}
var chart = c3.generate({
    size: {
        height: 600,
        width: 800
    },
    data: {
        xs: {''' + xs_str + '''},
        columns: [''' + columns_str + '''],
        type: 'scatter'
    },
    color: {
        pattern: [''' + color_list + ''']
    },
    tooltip: {
        contents: function (d, defaultTitleFormat, defaultValueFormat, color) {
        var $$ = this, config = $$.config,
            titleFormat = config.tooltip_format_title || defaultTitleFormat,
            nameFormat = config.tooltip_format_name || function (name) { return name; },
            valueFormat = config.tooltip_format_value || defaultValueFormat,
            text, i, title, value, name, bgcolor;
            text = "<table class='c3-tooltip c3-tooltip-container'>";
            for (i = 0; i < d.length; i++) {
                if (! (d[i] && (d[i].value || d[i].value === 0))) { continue; }
                value = d[i].x.toFixed(4) + ', ' + d[i].value.toFixed(4);
                name = nameFormat(d[i].name, d[i].ratio, d[i].id, d[i].index);
                bgcolor = $$.levelColor ? $$.levelColor(d[i].value) : color(d[i].id);

                text += "<tr class='" + d[i].id + "'>";
                text += "<td class='name'><span style='background-color:" + bgcolor + "'></span>" + name + "</td>";
                text += "<td class='value'>" + value + "</td>";
                text += "</tr>";
                text += "<tr style=><th colspan='2'><img border=0px src=" + imagepath[d[i].id][d[i].index] + "></th></tr>";
            }
            return text + "</table>";
        }
    },
    axis: {
        x: {
            label: 'tsne1',
            min: ''' +str(math.floor(x_min))+ ''',
            max: ''' +str(math.ceil(x_max))+ ''',
            tick: {
                fit: false
            }
        },
        y: {
            label: 'tsne2',
            min: ''' +str(math.floor(y_min))+ ''',
            max: ''' +str(math.ceil(y_max))+ ''',
            tick: {
                fit: false
            }
        }
    },
    legend: {
        item: {
            onclick: function (d) {
                if (event.ctrlKey) {
                    chart.toggle();
                }
                chart.toggle(d);
            }
        }
    }
});
''')
    with div():
        attr(cls='body')
        p('Lorem ipsum..')

    #print(doc)
    with open(args.out, 'w') as fo:
        fo.write(doc.render())



parser = argparse.ArgumentParser(
    description='Learning convnet from MINC-2500 dataset')
parser.add_argument('infile', help='Path to image-label-x-y file')
parser.add_argument('--categories', '-c', default='categories.txt',
                    help='Path to category list file')
parser.add_argument('--root', '-R', default='.',
                    help='Root directory path of image files')
parser.add_argument('--out', '-o', default='scatter.html',
                    help='Output directory')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
