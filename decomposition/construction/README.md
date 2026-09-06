# construction

Registry 构造 Pipeline，component_factory 构造具体组件。__init__.py 不自动导入 Registry，避免经 Pipeline 回到本包时发生循环导入。
