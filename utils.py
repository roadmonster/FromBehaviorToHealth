import io
import os

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class ReportBuilder():
    """
    Build a PDF report on given contents. Can accept text strings and matplotlib figures.
    """
    def __init__(
        self, contents, title, filename,
        subtitle=None, pagesize=letter, dpi=300,
        margins=[inch/2, inch/2, inch/4, inch/4]
    ):
        self._contents = contents
        self._title = title
        self._subtitle = subtitle
        self._filename = filename
        self._pagesize = pagesize
        self._dpi = dpi
        self._margins = margins
        self._bufpool = []
        self._story = []
        self._styles = getSampleStyleSheet()

    def _make_dir(self, filename):
        if '/' in filename:
            filepath = filename[:filename.rfind('/')]
            if len(filepath) > 0 and not os.path.exists(filepath):
                os.makedirs(filepath)

    def add_image(self, fig, size=(5, 3)):
        """Add image to story."""
        # Build image buffer
        buf = io.BytesIO()
        fig.set_size_inches(size[0], size[1])
        fig.savefig(buf, format='png', dpi=self._dpi)
        buf.seek(0)
        self._bufpool.append(buf)

        # Build image
        im = Image(buf, size[0] * inch, size[1] * inch)
        self._story.append(im)

    def add_text(
        self, text,
        style="Normal", fontsize=12,
        spacing_top=True, spacing_bot=True
    ):
        """Add text to story."""
        if spacing_top:
            self._story.append(Spacer(1, 12))

        text = text.replace('\n', '<br />\n')
        ptext = "<font size={}>{}</font>".format(fontsize, text)
        para = Paragraph(ptext, self._styles[style])
        self._story.append(para)

        if spacing_bot:
            self._story.append(Spacer(1, 12))

    def build_report(self):
        """Build story to final PDF."""
        # Create dir if not exist
        self._make_dir(self._filename)

        # Initialize document
        doc = SimpleDocTemplate(
            self._filename,
            pagesize=self._pagesize,
            leftMargin=self._margins[0], rightMargin=self._margins[1],
            topMargin=self._margins[2], bottomMargin=self._margins[3]
        )

        # Add title and subtitle
        has_title = self._title and len(self._title) > 0
        has_subtitle = self._subtitle and len(self._subtitle) > 0

        if has_title:
            self.add_text(
                self._title, style="Heading2", fontsize=20,
                spacing_bot=(not has_subtitle)
            )

        if has_subtitle:
            self.add_text(
                self._subtitle, style="Heading4",
                fontsize=14, spacing_top=(not has_title)
            )

        # Add contents
        for content in self._contents:
            if isinstance(content, tuple) and len(content) == 2 and isinstance(content[0], Figure):
                self.add_image(content[0], size=content[1])
            elif isinstance(content, Figure):
                self.add_image(content)
            elif isinstance(content, str):
                self.add_text(content)
            else:
                raise ValueError('Unknown Type: {}'.format(type(content)))

        # Build
        doc.build(self._story)

        # Release image buffers
        for buf in self._bufpool:
            buf.close()


class ColumnCleaner:
    """Apply data clean steps on a pandas Series."""
    def __init__(self, col, steps):
        self._col_origin = col.copy()
        self._col_cleaned = col.copy()
        self._steps = steps
        self._exec_clean()

    def get_clean_report(self):
        if len(self._steps) == 0:
            step_descriptions = ['No cleaning is performed.']
        else:
            step_descriptions = [
                '[{:d}] {}'.format(i + 1, step[2])
                for i, step in enumerate(self._steps)
            ]
        return '\n'.join(['Cleaning Steps:'] + step_descriptions)

    def _exec_step(self, step):
        idx = step[0](self._col_cleaned)
        self._col_cleaned.loc[idx] = step[1](self._col_cleaned.loc[idx])

    def _exec_clean(self):
        for step in self._steps:
            self._exec_step(step)

    def get_cleaned_column(self):
        return self._col_cleaned.copy()


class ColumnVisualizer:
    """Make a visualization on given pandas Series."""
    def __init__(self, col, col_type='infer'):
        self._col = col.copy()
        if col_type == 'infer':
            self._col_type = ColumnVisualizer.infer_type(self._col)
        else:
            self._col_type = col_type

    @staticmethod
    def infer_type(col):
        if len(col.unique()) > 30:
            return 'Continuous'
        else:
            return 'Categorical'

    def make_continuous_fig(self, title):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(title)
        fig.tight_layout()

        # Boxplot
        sns.boxplot(self._col.dropna(), ax=ax1)
        ax1.set_xlabel('')

        # Distribution Plot with mean and mean +- 3std
        mean = self._col.dropna().mean()
        std = self._col.dropna().std()
        sns.distplot(self._col.dropna(), ax=ax2)
        ax2.set_xlabel('')
        ax2.axvline(mean, color='r')
        if mean + 3 * std < self._col.max():
            ax2.axvline(mean + 3 * std, color='k', linestyle='dashed', linewidth=1)
        if mean - 3 * std > self._col.min():
            ax2.axvline(mean - 3 * std, color='k', linestyle='dashed', linewidth=1)

        # Annotate mean and std
        annotation = 'mean = {:.4f}\nstd = {:.4f}'.format(mean, std)
        ax2.annotate(
            annotation, xy=(0.95, 0.95),
            xycoords='axes fraction', fontsize=10,
            horizontalalignment='right', verticalalignment='top'
        )

        return fig

    def make_categorical_fig(self, title):
        fig, ax = plt.subplots()
        fig.suptitle(title)
        fig.tight_layout()

        # Count each value including NA
        data = self._col.value_counts().sort_index()
        # data.index = data.index.astype(str)
        # data = data.sort_index()

        # Build count plot
        sns.barplot(y=data.index, x=data, orient="h", ax=ax)
        ax.set_xlabel('')
        for i, bar in enumerate(ax.patches):
            label = ax.annotate(
                '{:d} ({:.2f}%)'.format(data.iloc[i], data.iloc[i] * 100 / data.sum()),
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                va='center', xytext=(5, 0), textcoords='offset points'
            )

            # Adjust plot size to fit annotation
            bbox = label.get_window_extent(fig.canvas.get_renderer())
            bbox_data = bbox.transformed(ax.transData.inverted())
            ax.update_datalim(bbox_data.corners())
            ax.autoscale_view()
        return fig

    def get_figure(self, title=''):
        if self._col_type == 'Continuous':
            return self.make_continuous_fig(title)
        else:
            return self.make_categorical_fig(title)


def clean_and_report(
    df_in, var_name, clean_steps,
    var_description=None, pdf_filename=None, col_type='infer',
    fig_size='auto', additional_reports=None
):
    # Make copy of input df
    df = df_in.copy()

    # List for PDF contents
    pdf_contents = []

    # Infer column type
    if col_type == 'infer':
        col_type = ColumnVisualizer.infer_type(df[var_name])

    # Get figure size if auto
    if fig_size == 'auto':
        if col_type == 'Continuous':
            fig_size = (9, 3)
        elif col_type == 'Categorical':
            fig_size = (5, 3)

    # Figure before cleaning
    fig_before = ColumnVisualizer(df[var_name], col_type=col_type).get_figure()
    pdf_contents.append((fig_before, fig_size))

    # Has no cleaning steps
    if len(clean_steps) == 0:
        pdf_contents.append('Cleaning is not performed on this variable.')

    # Has cleaning steps
    else:
        # Cleaning
        cleaner = ColumnCleaner(df[var_name], clean_steps)
        df[var_name] = cleaner.get_cleaned_column()
        clean_report = cleaner.get_clean_report()
        print(clean_report)
        pdf_contents.append(clean_report)

        # Figure after cleaning
        fig_after = ColumnVisualizer(df[var_name], col_type=col_type).get_figure()
        pdf_contents.append((fig_after, fig_size))

    # Missing number report
    missing_count = df[var_name].isna().sum()
    missing_report = 'There are {:d} ({:.2f}%) missing records.'.format(
        missing_count, missing_count / len(df) * 100
    )
    print(missing_report)
    pdf_contents.append(missing_report)

    # Has additional reports
    if additional_reports:
        pdf_contents += additional_reports

    # Build report
    if not pdf_filename:
        pdf_filename = './EDA_reports/{}.pdf'.format(var_name)

    ReportBuilder(
        pdf_contents, title=var_name,
        subtitle=var_description, filename=pdf_filename
    ).build_report()

    return df
